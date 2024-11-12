import string
import itertools
import laspy
import numpy as np
import pandas as pd
import torch
import os
import threading
import glob

from abc import ABC
from scipy import ndimage
from torch_geometric.data import Dataset, DataLoader, Data
from io import ply_io, pcd_io


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
        self.filenames = glob.glob(os.path.join(root_dir, '*.npy'))
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
        pos = torch.from_numpy(pos.copy()).type(torch.float).to(self.device).requires_grad_(False)

        # Place sample at origin
        local_shift = torch.round(torch.mean(pos[:, :3], axis=0)).requires_grad_(False)
        pos -= local_shift
        return Data(pos=pos, x=None, local_shift=local_shift)  # create Data object


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
        pc = np.vstack((in_file.x, in_file.y, in_file.z))
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


def make_dtm(pc, terrain_class):
    """
    Create a digital terrain model (DTM) from the point cloud.
    :param pc: (DataFrame) Point cloud DataFrame containing the terrain data.
    :param terrain_class: (int) Class label for terrain.
    :return:
        DataFrame: Updated point cloud with normalized heights.
    """
    grid_resolution = 0.5  # Resolution for voxelization
    pc = voxelise(pc, grid_resolution, z=False)  # Voxelize the point cloud

    # Identify ground points based on the terrain class
    ground = pc.loc[pc.label == terrain_class]
    ground['zmin'] = ground.groupby('VX').z.transform(np.median)  # get median height
    ground = ground[ground.z == ground.zmin].drop_duplicates('VX')  # keep lowest points

    # Create a mesh grid for the ground
    X, Y = np.meshgrid(
        np.arange(pc.xx.min(), pc.xx.max() + grid_resolution, grid_resolution),
        np.arange(pc.yy.min(), pc.yy.max() + grid_resolution, grid_resolution))

    ground_arr = pd.DataFrame(data=np.vstack([X.flatten(), Y.flatten()]).T, columns=['xx', 'yy'])
    VX_map = pc.loc[~pc.VX.duplicated()][['xx', 'yy', 'VX']]
    ground_arr = ground_arr.merge(VX_map, on=['xx', 'yy'], how='outer')  # map VX to ground_arr
    ground_arr = ground_arr.merge(ground[['z', 'VX']], on=['VX'], how='right')  # map z to ground_arr
    ground_arr.sort_values(['xx', 'yy'], inplace=True)

    # Fill NaN Values in ground_arr using a median filter
    ground_arr['ZZ'] = np.nan
    size = 3
    while np.any(np.isnan(ground_arr.ZZ)):
        ground_arr['ZZ'] = ndimage.generic_filter(
            ground_arr.z.values.reshape(*X.shape),
            lambda z: np.nanmedian(z),
            size=size
        ).flatten()
        size += 2  # increase the filter size

    # Update point cloud with normalized heights
    MAP = ground_arr.set_index('VX').ZZ.to_dict()
    pc['n_z'] = pc.z - pc.VX.map(MAP)

    return pc


def save_file(filename, pointcloud, additional_fields=[], verbose=False):
    #     if pointcloud.shape[0] == 0:
    #         print(filename, 'is empty...')
    #     else:
    if verbose:
        print('Saving file:', filename)

    cols = ['x', 'y', 'z'] + additional_fields

    if filename.endswith('.las'):
        las = laspy.create(file_version="1.4", point_format=7)
        las.header.offsets = np.min(pointcloud[:, :3], axis=0)
        las.header.scales = [0.001, 0.001, 0.001]

        las.x = pointcloud[:, 0]
        las.y = pointcloud[:, 1]
        las.z = pointcloud[:, 2]

        if len(additional_fields) != 0:
            additional_fields = additional_fields[3:]

            #  The reverse step below just puts the headings in the preferred order. They are backwards without it.
            col_idxs = list(range(3, pointcloud.shape[1]))
            additional_fields.reverse()

            col_idxs.reverse()
            for header, i in zip(additional_fields, col_idxs):
                column = pointcloud[:, i]
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


def save_pts(pc, box_dims, min_points_per_box, max_points_per_box, I, bx, by, bz, working_dir):
    """
    Save points in bounding boxes to files.
    :param pc: (DataFrame) Point cloud DataFrame.
    :param box_dims: (array) Dimensions of the box.
    :param min_points_per_box: (int) Minimum number of points to save.
    :param max_points_per_box: (int) Maximum number of points to save.
    :param I: (int) Index for naming the output file.
    :param bx: (float) Base X coordinate for the box.
    :param by: (float) Base Y coordinate for the box.
    :param bz: (float) Base Z coordinate for the box.
    :param working_dir: (str) Directory to save the output files.
    """
    pc = pc.loc[(pc.x.between(bx, bx + box_dims[0])) &
                (pc.y.between(by, by + box_dims[0])) &
                (pc.z.between(bz, bz + box_dims[0]))]
    if len(pc) > min_points_per_box:
        if len(pc) > max_points_per_box:
            pc = pc.sample(n=max_points_per_box)  # randomly sample points if too many
        np.save(os.path.join(working_dir, f'{I:07}'), pc[['x', 'y', 'z']].values)


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
