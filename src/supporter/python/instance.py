import datetime
import argparse
from tqdm.auto import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import networkx as nx
from separation_tools import *
from fit_cylinders import RANSAC_helper
import warnings

# Suppress warnings and chained assignment warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


def generate_path(samples, origins, n_neighbours=200, max_length=0):
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

    # Create graph and compute shortest paths
    G = nx.from_pandas_edgelist(edges, edge_attr=['length'])
    distance, shortest_path = nx.multi_source_dijkstra(G, sources=origins, weight='length')

    # Prepare paths DataFrame
    paths = pd.DataFrame(distance.items(), columns=['clstr', 'distance'])
    paths['base'] = params.not_base
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


def main():
    """Main function to execute the script."""
    start_time = datetime.datetime.now()

    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile', '-t', type=str, required=True, help='fsct directory')
    parser.add_argument('--odir', '-o', type=str, required=True, help='output directory')
    parser.add_argument('--tindex', type=str, required=True, help='path to tile index')
    parser.add_argument('--n-tiles', default=3, type=int, help='number of tiles (e.g., 3x3 or 5x5)')
    parser.add_argument('--overlap', default=False, type=float, help='buffer for adjacent tiles')
    parser.add_argument('--slice-thickness', default=0.2, type=float, help='slice thickness for constructing graph')
    parser.add_argument('--find-stems-height', default=1.5, type=float, help='height for identifying stems')
    parser.add_argument('--find-stems-thickness', default=0.5, type=float, help='thickness for identifying stems')
    parser.add_argument('--find-stems-min-radius', default=0.025, type=float, help='minimum radius of found stems')
    parser.add_argument('--find-stems-min-points', default=200, type=int, help='minimum points for found stems')
    parser.add_argument('--graph-edge-length', default=1, type=float, help='max distance to connect points in graph')
    parser.add_argument('--graph-maximum-cumulative-gap', default=np.inf, type=float,
                        help='max gap between base and cluster')
    parser.add_argument('--min-points-per-tree', default=0, type=int, help='min points for identified trees')
    parser.add_argument('--add-leaves', action='store_true', help='add leaf points')
    parser.add_argument('--add-leaves-voxel-length', default=0.5, type=float, help='voxel size for adding leaves')
    parser.add_argument('--add-leaves-edge-length', default=1, type=float,
                        help='max distance for leaf graph connections')
    parser.add_argument('--save-diameter-class', action='store_true', help='save into diameter class directories')
    parser.add_argument('--ignore-missing-tiles', action='store_true', help='ignore missing neighbouring tiles')
    parser.add_argument('--pandarallel', action='store_true', help='use pandarallel for parallel processing')
    parser.add_argument('--verbose', action='store_true', help='enable verbose output')

    params = parser.parse_args()

    # Initialize pandarallel if requested
    if params.pandarallel:
        try:
            from pandarallel import pandarallel
            pandarallel.initialize(progress_bar=params.verbose)
        except ImportError:
            print('--- pandarallel not installed ---')
            params.pandarallel = False

    # Print parameters if verbose
    if params.verbose:
        print('---- parameters ----')
        for k, v in params.__dict__.items():
            print(f'{k:<35}{v}')

    # Setup parameters
    params.not_base = -1
    params.xyz = ['x', 'y', 'z']  # shorthand for coordinate columns
    params.dir, params.fn = os.path.split(params.tile)
    params.n = int(params.fn.split('.')[0])
    params.pc = ply_io.read_ply(params.tile)
    params.pc['buffer'] = False
    params.pc['fn'] = params.n

    # Compute bounding box
    bbox = {
        'xmin': params.pc.x.min(),
        'xmax': params.pc.x.max(),
        'ymin': params.pc.y.min(),
        'ymax': params.pc.y.max(),
    }
    bbox = DictToClass(bbox)

    # Load neighboring tiles
    params.ti = pd.read_csv(params.tindex, sep=' ', names=['tile', 'x', 'y'])
    n_tiles = NearestNeighbors(n_neighbors=len(params.ti)).fit(params.ti[['x', 'y']])
    distances, indices = n_tiles.kneighbors(params.ti.loc[params.ti.tile == params.n][['x', 'y']])

    # Identify buffer tiles
    buffer_tiles = params.ti.loc[indices[0][1:params.n_tiles ** 2]]['tile'].values

    # Read in neighbouring tiles
    for i, t in tqdm(enumerate(buffer_tiles), total=len(buffer_tiles), desc='Read in neighbouring tiles',
                     disable=not params.verbose):
        try:
            b_tile = glob.glob(os.path.join(params.dir, f'{t:03}*.ply'))[0]
            tmp = ply_io.read_ply(b_tile)

            if params.overlap:
                tmp = tmp[(tmp.x.between(bbox.xmin - params.overlap, bbox.xmax + params.overlap)) &
                          (tmp.y.between(bbox.ymin - params.overlap, bbox.ymax + params.overlap))]
            if len(tmp) == 0:
                continue

            tmp['buffer'] = True
            tmp['fn'] = t
            params.pc = params.pc.append(tmp, ignore_index=True)
        except IndexError:
            path = os.path.join(params.dir, f'{t:03}*.ply')
            if params.ignore_missing_tiles:
                print(f'Tile {path} not available')
            else:
                raise Exception(f'Tile {path} not available')

    # Rename column if necessary
    if 'nz' in params.pc.columns:
        params.pc.rename(columns={'nz': 'n_z'}, inplace=True)

    # Optimize memory usage by selecting necessary columns
    params.pc = params.pc[['x', 'y', 'z', 'n_z', 'label', 'buffer', 'fn']]
    params.pc[['x', 'y', 'z', 'n_z']] = params.pc[['x', 'y', 'z', 'n_z']].astype(np.float32)
    params.pc[['label', 'fn']] = params.pc[['label', 'fn']].astype(np.int16)

    # Skeletonization
    if params.verbose:
        print('\n----- Skeletonization started -----')

    # Extract stem points
    stem_pc = params.pc[params.pc.label == 3]

    # Slice stem points
    stem_pc['slice'] = (stem_pc.z // params.slice_thickness).astype(int) * params.slice_thickness
    stem_pc['n_slice'] = (stem_pc.n_z // params.slice_thickness).astype(int)

    # Clustering within height slices
    stem_pc['clstr'] = -1
    label_offset = 0

    for slice_height in tqdm(np.sort(stem_pc.n_slice.unique()), disable=not params.verbose,
                             desc='Slice data vertically and clustering'):
        new_slice = stem_pc[stem_pc.n_slice == slice_height]

        if len(new_slice) > 200:
            dbscan = DBSCAN(eps=0.1, min_samples=20).fit(new_slice[params.xyz])
            new_slice['clstr'] = dbscan.labels_
            new_slice.loc[new_slice.clstr > -1, 'clstr'] += label_offset
            stem_pc.loc[new_slice.index, 'clstr'] = new_slice.clstr
            label_offset = stem_pc.clstr.max() + 1

    # Group skeleton points and fit convex hulls
    grouped = stem_pc[stem_pc.clstr != -1].groupby('clstr')
    if params.verbose:
        print('Fitting convex hulls to clusters')
    chull = grouped.apply(cube).reset_index(drop=True)

    # Identify possible stems
    if params.verbose:
        print('Identifying stems...')
    skeleton = grouped[params.xyz + ['n_z', 'n_slice', 'slice']].median().reset_index()
    skeleton['dbh_node'] = False

    find_stems_min = int(params.find_stems_height // params.slice_thickness)
    find_stems_max = int((params.find_stems_height + params.find_stems_thickness) // params.slice_thickness) + 1
    dbh_nodes_plus = skeleton[skeleton.n_slice.between(find_stems_min, find_stems_max)].clstr
    dbh_slice = stem_pc[stem_pc.clstr.isin(dbh_nodes_plus)]

    if len(dbh_slice) > 0:
        # Remove noise from dbh slice
        nn = NearestNeighbors(n_neighbors=10).fit(dbh_slice[params.xyz])
        distances, indices = nn.kneighbors()
        dbh_slice['nn'] = distances[:, 1:].mean(axis=1)
        dbh_slice = dbh_slice[dbh_slice.nn < dbh_slice.nn.quantile(q=0.9)]

        # DBSCAN to find potential stems
        dbscan = DBSCAN(eps=0.2, min_samples=50).fit(dbh_slice[['x', 'y']])
        dbh_slice['clstr_db'] = dbscan.labels_
        dbh_slice = dbh_slice[dbh_slice.clstr_db > -1]
        dbh_slice['cclstr'] = dbh_slice.groupby('clstr_db').clstr.transform('min')

        if len(dbh_slice) > 10:
            # RANSAC cylinder fitting
            if params.verbose:
                print('Fitting cylinders to possible stems...')
            dbh_cylinder = dbh_slice.groupby('cclstr').apply(RANSAC_helper, 100).to_dict()
            dbh_cylinder = pd.DataFrame(dbh_cylinder).T
            dbh_cylinder.columns = ['radius', 'centre', 'CV', 'cnt']
            dbh_cylinder[['x', 'y', 'z']] = pd.DataFrame(dbh_cylinder.centre.tolist(), index=dbh_cylinder.index)
            dbh_cylinder = dbh_cylinder.drop(columns=['centre']).astype(float)

            # Identify nodes based on cylinder properties
            skeleton.loc[skeleton.clstr.isin(dbh_cylinder.loc[
                                                 (dbh_cylinder.radius > params.find_stems_min_radius) &
                                                 (dbh_cylinder.cnt > params.find_stems_min_points) &
                                                 (dbh_cylinder.CV <= 0.15)
                                                 ].index.values), 'dbh_node'] = True

    in_tile_stem_nodes = skeleton[
        (skeleton.dbh_node) &
        (skeleton.x.between(bbox.xmin, bbox.xmax)) &
        (skeleton.y.between(bbox.ymin, bbox.ymax))
        ].clstr

    # Generate paths through all stem points
    if params.verbose:
        print('Generating graph, this may take a while...')
    wood_paths = generate_path(
        chull,
        skeleton[skeleton.dbh_node].clstr,
        n_neighbours=200,
        max_length=params.graph_edge_length
    )

    # Filter paths
    wood_paths = wood_paths.sort_values(['clstr', 'distance'])
    wood_paths = wood_paths[~wood_paths['clstr'].duplicated()]
    wood_paths = wood_paths[wood_paths.distance <= params.graph_maximum_cumulative_gap]

    if params.verbose:
        print('Merging skeleton points with graph')
    stems = pd.merge(skeleton, wood_paths, on='clstr', how='left')

    # Generate unique RGB for each stem
    unique_stems = stems.t_clstr.unique()
    RGB = pd.DataFrame({
        't_clstr': unique_stems,
        'red': np.random.randint(0, 255, size=len(unique_stems)),
        'green': np.random.randint(0, 255, size=len(unique_stems)),
        'blue': np.random.randint(0, 255, size=len(unique_stems))
    })
    RGB.loc[RGB.t_clstr == params.not_base, ['red', 'green', 'blue']] = [211, 211, 211]  # Grey for unassigned points
    stems = pd.merge(stems, RGB, on='t_clstr', how='right')

    # Read in all "stems" tiles and assign all stem points to a tree
    trees = pd.merge(
        stem_pc,
        stems[['clstr', 't_clstr', 'distance', 'red', 'green', 'blue']],
        on='clstr'
    )
    trees['cnt'] = trees.groupby('t_clstr').t_clstr.transform('count')
    trees = trees[trees.cnt > params.min_points_per_tree]
    in_tile_stem_nodes = trees[trees.t_clstr.isin(in_tile_stem_nodes)].t_clstr.unique()

    # Write out all trees
    params.base_I, I = {}, 0
    for i, b in tqdm(enumerate(dbh_cylinder.loc[in_tile_stem_nodes].sort_values('radius', ascending=False).index),
                     total=len(in_tile_stem_nodes),
                     desc='Writing stems to file',
                     disable=not params.verbose):

        if b == params.not_base:
            continue

        # Save based on diameter class if requested
        if params.save_diameter_class:
            d_dir = f'{(dbh_cylinder.loc[b].radius * 2 // 0.1) / 10:.1f}'
            os.makedirs(os.path.join(params.odir, d_dir), exist_ok=True)
            ply_io.write_ply(os.path.join(params.odir, d_dir, f'{params.n:03}_T{I}.leafoff.ply'),
                             trees[trees.t_clstr == b])
        else:
            ply_io.write_ply(os.path.join(params.odir, f'{params.n:03}_T{I}.leafoff.ply'),
                             trees[trees.t_clstr == b])
        params.base_I[b] = I
        I += 1

    # Add leaves if requested
    if params.add_leaves:
        if params.verbose:
            print('Adding leaves to stems, this may take a while...')

        # Link stem numbers to clusters
        stem2tlsctr = stems[stems.t_clstr != params.not_base][['clstr', 't_clstr']].set_index('clstr').to_dict()[
            't_clstr']
        chull['stem'] = chull.clstr.map(stem2tlsctr)

        # Identify unlabelled woody points to add back to leaves
        unlabelled_wood = chull[chull.stem.isna()]
        unlabelled_wood = stem_pc[stem_pc.clstr.isin(unlabelled_wood.clstr.tolist() + [-1])]
        unlabelled_wood = unlabelled_wood[unlabelled_wood.n_z >= 2]

        # Extract wood points attributed to a base and the last cluster of the graph (i.e. a tip)
        is_tip = wood_paths.set_index('clstr')['is_tip'].to_dict()
        chull = chull[chull.stem.notna()]
        chull['is_tip'] = chull.clstr.map(is_tip)
        chull = chull[(chull.is_tip) & (chull.n_z > params.find_stems_height)]
        chull['xlabel'] = 2

        # Process leaf points
        lvs = params.pc[(params.pc.label == 1) & (params.pc.n_z >= 2)].copy()
        lvs = lvs.append(unlabelled_wood, ignore_index=True)
        lvs.reset_index(inplace=True)

        # Voxelise leaf points
        lvs = voxelise(lvs, length=params.add_leaves_voxel_length)
        lvs_gb = lvs.groupby('VX')[params.xyz]
        lvs_min = lvs_gb.min()
        lvs_max = lvs_gb.max()
        lvs_med = lvs_gb.median()

        # Create corners for leaf voxels
        cnrs = np.vstack([lvs_min.x, lvs_med.y, lvs_med.z]).T
        clstr = np.tile(np.arange(len(lvs_min.index)) + 1 + chull.clstr.max(), 6)
        VX = np.tile(lvs_min.index, 6)
        cnrs = np.vstack([cnrs, np.vstack([lvs_max.x, lvs_med.y, lvs_med.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_min.y, lvs_med.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_med.y, lvs_min.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_min.x, lvs_max.y, lvs_max.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_max.x, lvs_max.y, lvs_min.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_max.y, lvs_med.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_min.y, lvs_max.z]).T])

        # Create DataFrame for leaf corners
        leaf_corners = pd.DataFrame(cnrs, columns=['x', 'y', 'z'])
        leaf_corners['clstr'] = clstr
        leaf_corners['VX'] = VX

        # Save leaf voxel data to output
        leaf_output_path = os.path.join(params.odir, f'{params.n:03}_leaves.ply')
        ply_io.write_ply(leaf_output_path, leaf_corners)

        # Finalize and clean up
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    if params.verbose:
        print(f'--- Processing completed in {duration} ---')


if __name__ == "__main__":
    main()