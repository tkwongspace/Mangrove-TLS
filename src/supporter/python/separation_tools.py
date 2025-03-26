import string
import itertools
import laspy
import numpy as np
import pandas as pd
import networkx as nx
import shutil
import torch
import os
import glob
import warnings
from abc import ABC

from sklearn.neighbors import NearestNeighbors
from scipy import ndimage
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from torch_geometric.data import Dataset, Data
from tqdm.auto import tqdm
from src.supporter.python.io import ply_io, pcd_io
from src.supporter.python.fit_cylinders import RANSAC_helper

class DictToClass:
    """
    Convert a dictionary to a class for easier attribute access.
    """
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


class TestingDataset(Dataset, ABC):
    """
    Custom PyTorch Dataset for loading point cloud data from .npy files.
    """
    def __init__(self, root_dir, points_per_box, device):
        super(TestingDataset, self).__init__()  # Initialize the parent class
        self.filenames = glob.glob(os.path.join(root_dir, '*.npy'))  # Load all npy files
        self.device = device
        self.points_per_box = points_per_box

    def __len__(self):
        return len(self.filenames)  # number of files in the dataset

    def __getitem__(self, index):
        """
        Load a point cloud from a file and process it.
        :param index: (int) Index of the file to load.
        :return:
            Data: A PyTorch geometric Data object containing the point cloud.
        """
        point_cloud = np.load(self.filenames[index])  # load the point cloud
        pos = point_cloud[:, :3]
        pos = torch.from_numpy(pos.copy()).float().to(self.device).requires_grad_(False)

        # Center the sample at the origin
        local_shift = torch.round(torch.mean(pos[:, :3], axis=0)).requires_grad_(False)
        pos -= local_shift

        # # Ensure consistent size for pos [Remember to remove those padding points later]
        # if pos.shape[1] < self.points_per_box:
        #     # Pad the data with zeros
        #     padding = torch.zeros((3, self.points_per_box - pos.shape[1])).to(self.device)
        #     pos = torch.cat((pos, padding), dim=1)

        return Data(pos=pos, x=None, local_shift=local_shift)  # create Data object

    def len(self): pass
    def get(self): pass


def compute_bbox(pc):
    """
    Compute the bounding box of the point cloud.
    :param pc: (DataFrame) Point cloud DataFrame with 'x', 'y', 'z' columns.
    :return:
        An object containing min and max coordinates of the bounding box.
    """
    bbox_min = pc.min().to_dict()
    bbox_min = {k + 'min':v for k, v in bbox_min.items()}
    bbox_max = pc.max().to_dict()
    bbox_max = {k + 'max':v for k, v in bbox_max.items()}
    return DictToClass({**bbox_min, **bbox_max})


def compute_plot_centre(pc):
    """
    Calculate the center of the plot.
    :param pc: (DataFrame) Point cloud DataFrame with 'x', 'y' coordinates.
    :return:
        np.ndarray: The center coordinates of the plot.
    """
    plot_min, plot_max = pc[['x', 'y']].min(), pc[['x', 'y']].max()
    return (plot_min + ((plot_max - plot_min) / 2)).values


def cube(pc):
    """Return a random sample of points forming the convex hull."""
    try:
        if len(pc) > 5:
            vertices = ConvexHull(pc[['x', 'y', 'z']]).vertices
            idx = np.random.choice(vertices, size=len(vertices), replace=False)
            return pc.loc[pc.index[idx]]
    except Exception as e:
        print(f"Error in cube function: {e}")
    return pc


def downsample(pc, v_length, accurate=False, keep_columns=None, keep_points=False, voxel_method='random',
               return_vx=False, verbose=False):
    """
    Downsample the point cloud using voxel grid filtering.
    :param pc: (DataFrame) Input point cloud.
    :param v_length: (float) Voxel length for downsampling.
    :param accurate: (bool) If True, uses median point for downsampling.
    :param keep_columns: (list) Additional columns to keep in the output.
    :param keep_points: (bool) If True, returns all points, else returns downsampled points.
    :param voxel_method: (str) Method for voxelization ('random' or 'bytes').
    :param return_vx: (bool) If True, includes 'VX' column in the output.
    :param verbose: (bool) If True, prints additional information.
    :return:
        DataFrame: Downsampled point cloud.
    """
    pc = pc.drop(columns=[c for c in ['downsample', 'VX'] if c in pc.columns])
    columns = pc.columns.to_list() if keep_columns is None else pc.columns.to_list() + keep_columns
    if return_vx: columns += ['VX']

    pc = voxelise(pc, v_length, method=voxel_method)

    if accurate:
        # to find central point (closest to median)
        group_by = pc.groupby('VX')
        pc.loc[:, 'mx'] = group_by.x.transform(np.median)
        pc.loc[:, 'my'] = group_by.y.transform(np.median)
        pc.loc[:, 'mz'] = group_by.z.transform(np.median)
        pc.loc[:, 'dist'] = np.linalg.norm(pc[['x', 'y', 'z']].to_numpy(dtype=np.float32) -
                                           pc[['mx', 'my', 'mz']].to_numpy(dtype=np.float32), axis=1)
        pc.loc[:, 'downsample'] = False
        pc.loc[~pc.sort_values(['VX', 'dist']).duplicated('VX'), 'downsample'] = True

    else:
        pc.loc[:, 'downsample'] = False
        pc.loc[~pc.VX.duplicated(), 'downsample'] = True

    if keep_points:
        return pc[columns + ['downsample']]
    else:
        return pc.loc[pc.downsample][columns]


def find_stems(skl, slices, params=None, find_stems_min_radius=0.025, find_stems_min_points=200,
               pandarallel=False, verbose=True):
    # Initial settings
    if params is not None:
        verbose = params.verbose
        find_stems_min_radius = params.find_stems_min_radius
        find_stems_min_points = params.find_stems_min_points
        pandarallel = params.pandarallel

    # Remove noise from dbh slice
    nn = NearestNeighbors(n_neighbors=10).fit(slices[['x', 'y', 'z']])
    distances, indices = nn.kneighbors()
    slices.loc[:, 'nn'] = distances[:, 1:].mean(axis=1)
    slices = slices.loc[slices.nn < slices.nn.quantile(q=0.9)].copy()

    # DBSCAN to find potential stems
    dbscan = DBSCAN(eps=0.2, min_samples=50).fit(slices[['x', 'y']])
    slices['clstr_db'] = dbscan.labels_
    slices = slices.loc[slices.clstr_db > -1].copy()
    slices.loc[:, 'cclstr'] = slices.groupby('clstr_db').clstr.transform('min')

    if len(slices) > 10:
        # RANSAC cylinder fitting
        if pandarallel:
            if verbose:
                print('>> Fitting cylinders to possible stems...')
            fitted_cyl = slices.groupby('cclstr').parallel_apply(RANSAC_helper, 100).to_dict()
        else:
            group_xyz = slices.groupby('cclstr')
            fitted_cyl = {}
            for name, xyz in tqdm(group_xyz, desc="Fitting cylinders to possible stems", disable=not verbose):
                result = RANSAC_helper(xyz, 100)
                fitted_cyl[name] = result

        # extract coordinates of cylinder center to table
        fitted_cyl = pd.DataFrame(fitted_cyl).T
        fitted_cyl.columns = ['radius', 'centre', 'CV', 'cnt']
        fitted_cyl.loc[:, ['x', 'y', 'z']] = pd.DataFrame(fitted_cyl.centre.tolist(), index=fitted_cyl.index)
        fitted_cyl = fitted_cyl.drop(columns=['centre']).astype(float)

        # Identify nodes based on cylinder properties
        skl.loc[skl.clstr.isin(fitted_cyl.loc[
             (fitted_cyl.radius > find_stems_min_radius) &
             (fitted_cyl.cnt > find_stems_min_points) &
             (fitted_cyl.CV <= 0.15)
         ].index.values), 'dbh_node'] = True

    else:
        warnings.warn(f'No cylinder found with radius {find_stems_min_radius}. Set cylinders to None.')
        fitted_cyl = None

    return skl, fitted_cyl


def generate_path(samples, origins, n_neighbours=200, max_length=0, params=None):
    """Generate paths through the point cloud based on nearest neighbors."""
    nn = NearestNeighbors(n_neighbors=n_neighbours).fit(samples[['x', 'y', 'z']])
    distances, indices = nn.kneighbors()

    from_to_all = pd.DataFrame(np.vstack([
        np.repeat(samples.clstr.values, n_neighbours),
        samples.iloc[indices.ravel()].clstr.values,
        distances.ravel()
    ]).T, columns=['source', 'target', 'length'])

    # Remove X-X connections and keep edges with minimum distance
    from_to_all = from_to_all[from_to_all.target != from_to_all.source]
    edges = from_to_all.groupby(['source', 'target']).length.min().reset_index()
    edges = edges[edges.length <= max_length]

    # Filter origins based on edges
    origins = [s for s in origins if s in edges.source.values]

    # Create graph and compute the shortest paths
    G = nx.from_pandas_edgelist(edges, edge_attr=['length'])
    distance, shortest_path = nx.multi_source_dijkstra(G, sources=origins, weight='length')

    # Prepare paths DataFrame
    paths = pd.DataFrame(distance.items(), columns=['clstr', 'distance'])
    paths['base'] = -1 if params is None else params.not_base
    for p in paths.index:
        paths.at[p, 'base'] = shortest_path[p][0]
    paths.columns = ['clstr', 'distance', 't_clstr']

    # Identify branch tips
    node_occurance = {}
    for path in shortest_path.values():
        for node in path:
            node_occurance[node] = node_occurance.get(node, 0) + 1

    tips = [k for k, v in node_occurance.items() if v == 1]
    paths['is_tip'] = paths.clstr.isin(tips)

    return paths


def load_file(filename, additional_headers=False, verbose=False):
    """
    Load a point cloud file.
    :param filename: (str) Path to the point cloud file.
    :param additional_headers: (bool) If True, returns additional columns.
    :param verbose: (bool) If True, prints information about the loaded file.
    :return:
        DataFrame: Loaded point cloud.
        list: Additional headers if requested.
    """
    file_extension = os.path.splitext(filename)[1]

    if file_extension in ['.las', '.laz']:
        in_file = laspy.read(filename)
        pc = np.vstack((in_file.x, in_file.y, in_file.z)).T
        pc = pd.DataFrame(data=pc, columns=['x', 'y', 'z'])
    elif file_extension == '.ply':
        pc = ply_io.read_ply(filename)
    elif file_extension == '.pcd':
        pc = pcd_io.read_pcd(filename)
    else:
        raise Exception('! Point cloud format not recognised:' + filename)

    if verbose:
        print(f'-- Read in {filename} with {len(pc)} points.')

    if additional_headers:
        return pc, [c for c in pc.columns if c not in ['x', 'y', 'z']]
    else:
        return pc


def make_dtm(params = None, pc=None, terrain_class = 0):
    """
    Create a digital terrain model (DTM) from the point cloud.
    :param params: (dict) Dictionary of parameters.
    :param pc: (DataFrame) Point cloud DataFrame containing the terrain data.
    :param terrain_class: (int) Class label for terrain.
    :return:
        DataFrame: Updated point cloud with normalized heights.
    """
    if params is None and pc is not None:
        grid_resolution = 0.5  # Resolution for voxelization
        pc = voxelise(pc, grid_resolution, z=False)  # Voxelize the point cloud

        # Identify ground points based on the terrain class
        ground = pc.loc[pc.label == terrain_class]
        ground.loc[:, 'zmin'] = ground.groupby('VX').z.transform(np.median)  # get median height
        ground = ground[ground.z == ground.zmin].drop_duplicates('VX')  # keep lowest points

        # Create a mesh grid for the ground
        X, Y = np.meshgrid(
            np.arange(pc.xx.min(), pc.xx.max() + grid_resolution, grid_resolution),
            np.arange(pc.yy.min(), pc.yy.max() + grid_resolution, grid_resolution))

        ground_arr = pd.DataFrame(data=np.vstack([X.flatten(), Y.flatten()]).T, columns=['xx', 'yy'])
        VX_map = pc.loc[~pc.VX.duplicated()][['xx', 'yy', 'VX']]
        ground_arr = ground_arr.merge(VX_map, on=['xx', 'yy'], how='outer')  # map VX to ground_arr
        ground_arr = ground_arr.merge(ground[['z', 'VX']], how='left', on=['VX'])  # map z to ground_arr
        ground_arr.sort_values(['xx', 'yy'], inplace=True)

        # Fill NaN Values in ground_arr using a median filter
        ground_arr['ZZ'] = np.nan
        size = 3
        while np.any(np.isnan(ground_arr.ZZ)):
            ground_arr['ZZ'] = ndimage.generic_filter(
                ground_arr.z.values.reshape(*X.shape), lambda z: np.nanmedian(z), size=size
            ).flatten()
            size += 2  # increase the filter size

        # Update point cloud with normalized heights
        MAP = ground_arr.set_index('VX').ZZ.to_dict()
        pc['n_z'] = pc.z - pc.VX.map(MAP)

        return pc

    elif params is not None and pc is None:
        params.grid_resolution = 0.5
        params.pc = voxelise(params.pc, params.grid_resolution, z=False)

        # Identify ground points based on the terrain class
        ground = params.pc.loc[params.pc.label == params.terrain_class]
        ground.loc[:, 'zmin'] = ground.groupby('VX').z.transform(np.median)  # get median height
        ground = ground[ground.z == ground.zmin].drop_duplicates('VX')  # keep lowest points

        # Create a mesh grid for the ground
        X, Y = np.meshgrid(
            np.arange(params.pc.xx.min(), params.pc.xx.max() + params.grid_resolution, params.grid_resolution),
            np.arange(params.pc.yy.min(), params.pc.yy.max() + params.grid_resolution, params.grid_resolution))

        ground_arr = pd.DataFrame(data=np.vstack([X.flatten(), Y.flatten()]).T, columns=['xx', 'yy'])
        VX_map = params.pc.loc[~params.pc.VX.duplicated()][['xx', 'yy', 'VX']]
        ground_arr = ground_arr.merge(VX_map, on=['xx', 'yy'], how='outer')  # map VX to ground_arr
        ground_arr = ground_arr.merge(ground[['z', 'VX']], how='left', on=['VX'])  # map z to ground_arr
        ground_arr.sort_values(['xx', 'yy'], inplace=True)

        # Fill NaN Values in ground_arr using a median filter
        ground_arr['ZZ'] = np.nan
        size = 3
        while np.any(np.isnan(ground_arr.ZZ)):
            ground_arr['ZZ'] = ndimage.generic_filter(
                ground_arr.z.values.reshape(*X.shape), lambda z: np.nanmedian(z), size=size
            ).flatten()
            size += 2  # increase the filter size

        # Update point cloud with normalized heights
        MAP = ground_arr.set_index('VX').ZZ.to_dict()
        params.pc['n_z'] = params.pc.z - params.pc.VX.map(MAP)

        return params
    else:
        raise Exception("! Either params or pc shall be input.")


def make_folder_structure(params):
    if params.out_dir is None:
        params.out_dir = os.path.join(params.directory, params.filename + '_FSCT_output')
    params.working_dir = os.path.join(params.out_dir, params.basename + '.tmp')

    if not os.path.isdir(params.out_dir):
        os.makedirs(params.out_dir)

    if not os.path.isdir(params.working_dir):
        os.makedirs(params.working_dir)
    else:
        shutil.rmtree(params.working_dir, ignore_errors=True)
        os.makedirs(params.working_dir)

    if params.verbose:
        print('-- Output directory:', params.out_dir)
        print('-- Scratch directory:', params.working_dir)

    return params


def save_file(filename, pointcloud, additional_fields=None, verbose=False):
    #     if pointcloud.shape[0] == 0:
    #         print(filename, 'is empty...')
    #     else:
    if verbose:
        print('Saving file:', filename)

    cols = ['x', 'y', 'z'] if additional_fields is None else ['x', 'y', 'z'] + additional_fields

    if filename.endswith('.las'):
        las = laspy.create(file_version="1.4", point_format=7)
        las.header.offsets = pointcloud[['x', 'y', 'z']].min(axis=0)
        las.header.scales = [0.001, 0.001, 0.001]

        las.x = pointcloud['x']
        las.y = pointcloud['y']
        las.z = pointcloud['z']

        if len(additional_fields) != 0:
            # additional_fields = additional_fields[3:]

            #  The reverse step below just puts the headings in the preferred order. They are backwards without it.
            # col_idxs = list(range(3, pointcloud.shape[1]))
            col_idxs = [pointcloud.columns.get_loc(c) for c in additional_fields]
            additional_fields.reverse()
            col_idxs.reverse()

            for header, i in zip(additional_fields, col_idxs):
                column = pointcloud.iloc[:, i]
                if header in ['red', 'green', 'blue']:
                    setattr(las, header, column)
                else:
                    las.add_extra_dim(laspy.ExtraBytesParams(name=header, type="f8"))
                    setattr(las, header, column)
        las.write(filename)
        if not verbose:
            print("Saved.")

    elif filename.endswith('.csv'):
        pd.DataFrame(pointcloud).to_csv(filename, header=None, index=None, sep=' ')
        print("Saved to:", filename)

    elif filename.endswith('.ply'):
        if not isinstance(pointcloud, pd.DataFrame):
            cols = list(set(cols))
            pointcloud = pd.DataFrame(pointcloud, columns=cols)
        ply_io.write_ply(filename, pointcloud[cols])
        print("Saved to:", filename)


def save_pts(params=None, pc=None, I=None, bx=None, by=None, bz=None,
             box_dims=None, min_points_per_box=None, max_points_per_box=None,
             working_dir='./'):
    """
    Save points in bounding boxes to files.
    :param params: (Dict) A dictionary of input parameters.
    :param pc: (DataFrame) Point cloud DataFrame.
    :param I: (int) Index for naming the output file.
    :param bx: (float) Base X coordinate for the box.
    :param by: (float) Base Y coordinate for the box.
    :param bz: (float) Base Z coordinate for the box.
    :param box_dims: (array) Dimensions of the box.
    :param min_points_per_box: (int) Minimum number of points to save.
    :param max_points_per_box: (int) Maximum number of points to save.
    :param working_dir: (str) Directory to save the output files.
    """
    if params is None and pc is not None:
        pc = pc.loc[(pc.x.between(bx, bx + box_dims[0])) &
                    (pc.y.between(by, by + box_dims[0])) &
                    (pc.z.between(bz, bz + box_dims[0]))]
        if len(pc) > min_points_per_box:
            if len(pc) > max_points_per_box:
                pc = pc.sample(n=max_points_per_box)  # randomly sample points if too many
            np.save(os.path.join(working_dir, f'{I:07}'), pc[['x', 'y', 'z']].values)
    elif params is not None and pc is None:
        pc = params.pc.loc[(params.pc.x.between(bx, bx + box_dims[0])) &
                           (params.pc.y.between(by, by + box_dims[0])) &
                           (params.pc.z.between(bz, bz + box_dims[0]))]
        if len(pc) > params.min_points_per_box:
            if len(pc) > params.max_points_per_box:
                pc = pc.sample(n=params.max_points_per_box)
            np.save(os.path.join(params.working_dir, f'{I:07}'), pc[['x', 'y', 'z']].values)
    else:
        raise Exception("! Either params or pc shall be input.")


def slice_cluster(pc, label_offset, verbose=True):
    for slice_height in tqdm(np.sort(pc.n_slice.unique()),
                             disable=verbose,
                             desc='Slice data vertically and clustering'):
        new_slice = pc.loc[pc.n_slice == slice_height]
        if len(new_slice) > 200:
            dbscan = DBSCAN(eps=0.1, min_samples=20).fit(new_slice[['x', 'y', 'z']])
            new_slice.loc[:, 'clstr'] = dbscan.labels_
            new_slice.loc[new_slice.clstr > -1, 'clstr'] += label_offset
            pc.loc[new_slice.index, 'clstr'] = new_slice.clstr
            label_offset = pc.clstr.max() + 1

    return pc


def voxelise(tmp, length, method='random', z=True):
    """
    Voxelize the point cloud data.
    :param tmp: (DataFrame) Input point cloud DataFrame.
    :param length: (float) Voxel length.
    :param method: (str) Method for generating voxel codes ('random' or 'bytes').
    :param z: (bool) If True, includes z-coordinates in voxelization.
    :return:
        DataFrame: Voxelized point cloud DataFrame with 'VX' column.
    """
    tmp.loc[:, 'xx'] = tmp.x // length * length
    tmp.loc[:, 'yy'] = tmp.y // length * length
    if z: tmp.loc[:, 'zz'] = tmp.z // length * length

    if method == 'random':
        code = lambda: ''.join(np.random.choice([x for x in string.ascii_letters], size=8))
        xD = {x: code() for x in tmp.xx.unique()}
        yD = {y: code() for y in tmp.yy.unique()}
        if z: zD = {z: code() for z in tmp.zz.unique()}
        tmp.loc[:, 'VX'] = tmp.xx.map(xD) + tmp.yy.map(yD)
        if z: tmp.VX += tmp.zz.map(zD)
    elif method == 'bytes':
        code = lambda row: np.array([row.xx, row.yy] + [row.zz] if z else []).tobytes()
        tmp.loc[:, 'VX'] = tmp.apply(code, axis=1)
    else:
        raise Exception('method {} not recognised: choose "random" or "bytes"')

    return tmp
