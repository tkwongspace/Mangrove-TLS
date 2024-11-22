import datetime
import argparse
import pickle
import warnings

from mpmath import identify
from tqdm.auto import tqdm

from supporter.python.fsct_initial_param import initial_parameters
from src.supporter.python.separation_tools import *
from src.supporter.python.build_skeleton import build_skeleton
from src.supporter.python.identify_stems import identify_stems

# Suppress warnings and chained assignment warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


def main():
    """Main function to execute the script."""
    start_time = datetime.datetime.now()

    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', default='', type=str, required=True,
                        help='Path to pickled parameter file')
    parser.add_argument('--odir', '-o', type=str, required=True,
                        help='output directory')
    parser.add_argument('--slice-thickness', default=0.2, type=float,
                        help='slice thickness for constructing graph')
    parser.add_argument('--find-stems-height', default=1.5, type=float,
                        help='height for identifying stems')
    parser.add_argument('--find-stems-thickness', default=0.5, type=float,
                        help='thickness for identifying stems')
    parser.add_argument('--find-stems-min-radius', default=0.025, type=float,
                        help='minimum radius of found stems')
    parser.add_argument('--find-stems-min-points', default=200, type=int,
                        help='minimum points for found stems')
    parser.add_argument('--graph-edge-length', default=1, type=float,
                        help='max distance to connect points in graph')
    parser.add_argument('--graph-maximum-cumulative-gap', default=np.inf, type=float,
                        help='max gap between base and cluster')
    parser.add_argument('--min-points-per-tree', default=0, type=int,
                        help='min points for identified trees')
    parser.add_argument('--add-leaves', action='store_true',
                        help='add leaf points')
    parser.add_argument('--add-leaves-voxel-length', default=0.5, type=float,
                        help='voxel size for adding leaves')
    parser.add_argument('--add-leaves-edge-length', default=1, type=float,
                        help='max distance for leaf graph connections')
    parser.add_argument('--save-diameter-class', action='store_true',
                        help='save into diameter class directories')
    parser.add_argument('--ignore-missing-tiles', action='store_true',
                        help='ignore missing neighbouring tiles')
    parser.add_argument('--pandarallel', action='store_true',
                        help='use pandarallel for parallel processing')
    parser.add_argument('--verbose', action='store_true',
                        help='enable verbose output')

    params = parser.parse_args()

    if os.path.isfile(params.params):
        p_space = pickle.load(open(params.params, 'rb'))
        for k, v in p_space.__dict__.items():
            # override initial parameters
            if k == 'params': continue
            setattr(params, k, v)
    else:
        for k, v in initial_parameters.items():
            setattr(params, k, v)

    # Initialize pandarallel if requested
    if params.pandarallel:
        try:
            from pandarallel import pandarallel
            pandarallel.initialize(progress_bar=params.verbose)
        except ImportError:
            print('>> Warning: pandarallel not installed.')
            params.pandarallel = False

    # Print parameters if verbose
    if params.verbose:
        print('---- parameters ----')
        for k, v in params.__dict__.items():
            print(f'{k:<35}{v}')

    # Setup parameters
    params.not_base = -1
    xyz = ['x', 'y', 'z']  # shorthand for coordinate columns

    # Compute bounding box
    bbox = {
        'xmin': params.pc.x.min(),
        'xmax': params.pc.x.max(),
        'ymin': params.pc.y.min(),
        'ymax': params.pc.y.max(),
    }
    bbox = DictToClass(bbox)

    # Rename column if necessary
    if 'nz' in params.pc.columns:
        params.pc.rename(columns={'nz': 'n_z'}, inplace=True)

    # Optimize memory usage by selecting necessary columns
    params.pc = params.pc[['x', 'y', 'z', 'n_z', 'label']]
    params.pc[['x', 'y', 'z', 'n_z']] = params.pc[['x', 'y', 'z', 'n_z']].astype(np.float32)

    # Build skeleton
    if params.verbose:
        print('>> Start building skeleton..')
    stem_pc, chull, grouped = build_skeleton(params)

    # Identify possible stems
    if params.verbose:
        print('>> Identifying stems...')
    skeleton, in_tile_stem_nodes, dbh_cylinder = identify_stems(params=params,
                                                                clusters=grouped,
                                                                stem=stem_pc,
                                                                bounding_box=bbox)

    # Generate paths through all stem points
    if params.verbose:
        print('>> Generating graph, this may take a while...')
    wood_paths = generate_path(chull,
                               skeleton[skeleton.dbh_node].clstr,
                               n_neighbours=200,
                               max_length=params.graph_edge_length)

    # Filter paths
    wood_paths = wood_paths.sort_values(['clstr', 'distance'])
    wood_paths = wood_paths[~wood_paths['clstr'].duplicated()]
    wood_paths = wood_paths[wood_paths.distance <= params.graph_maximum_cumulative_gap]

    if params.verbose:
        print('>> Merging skeleton points with graph')
    stems = pd.merge(skeleton, wood_paths, on='clstr', how='left')

    # Generate unique RGB for each stem (for visualization)
    unique_stems = stems.t_clstr.unique()
    rgb = pd.DataFrame({
        't_clstr': unique_stems,
        'red': np.random.randint(0, 255, size=len(unique_stems)),
        'green': np.random.randint(0, 255, size=len(unique_stems)),
        'blue': np.random.randint(0, 255, size=len(unique_stems))
    })
    rgb.loc[rgb.t_clstr == params.not_base, ['red', 'green', 'blue']] = [211, 211, 211]  # Grey for unassigned points
    # rgb.loc[rgb.t_clstr == params.not_base, :] = [np.nan, 211, 211, 211]
    stems = pd.merge(stems, rgb, on='t_clstr', how='right')

    # Read in all "stems" tiles and assign all stem points to a tree
    trees = pd.merge(stem_pc, stems[['clstr', 't_clstr', 'distance', 'red', 'green', 'blue']],
                     on='clstr')
    trees['cnt'] = trees.groupby('t_clstr').t_clstr.transform('count')
    trees = trees[trees.cnt > params.min_points_per_tree]
    in_tile_stem_nodes = trees.loc[trees.t_clstr.isin(in_tile_stem_nodes)].t_clstr.unique()

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
            save_path = os.path.join(params.odir, d_dir, f'{params.n:03}_T{I}.leafoff.las')
        else:
            save_path = os.path.join(params.odir, f'{params.n:03}_T{I}.leafoff.las')
        save_file(filename=save_path, pointcloud=trees[trees.t_clstr == b])
        params.base_I[b] = I
        I += 1

    # Add leaves if requested
    if params.add_leaves:
        if params.verbose:
            print('>> Adding leaves to stems, this may take a while...')

        # Link stem numbers to clusters
        stem_to_tls_ctr = stems[stems.t_clstr != params.not_base][['clstr', 't_clstr']]. \
            set_index('clstr').to_dict()['t_clstr']
        chull.loc[:, 'stem'] = chull.clstr.map(stem_to_tls_ctr)

        # Identify unlabelled woody points to add back to leaves
        unlabelled_wood = chull.loc[[chull.stem.isna()]]  # NEED ATTENTION
        unlabelled_wood = stem_pc.loc[stem_pc.clstr.isin(unlabelled_wood.clstr.tolist() + [-1])]
        unlabelled_wood = unlabelled_wood.loc[unlabelled_wood.n_z >= 2]

        # Extract wood points attributed to a base and the last cluster of the graph (i.e. a tip)
        is_tip = wood_paths.set_index('clstr')['is_tip'].to_dict()
        chull = chull.loc[[chull.stem.notna()]]  # NEED ATTENTION
        chull.loc[:, 'is_tip'] = chull.clstr.map(is_tip)
        chull = chull.loc[chull.is_tip & (chull.n_z > params.find_stems_height)]
        chull.loc[:, 'xlabel'] = 2

        # Process leaf points
        lvs = params.pc.loc[(params.pc.label == 1) & (params.pc.n_z >= 2)].copy()
        lvs = lvs.append(unlabelled_wood, ignore_index=True)
        lvs.reset_index(inplace=True)

        # Voxelise leaf points
        lvs = voxelise(lvs, length=params.add_leaves_voxel_length)
        lvs_gb = lvs.groupby('VX')[xyz]
        lvs_min = lvs_gb.min()
        lvs_max = lvs_gb.max()
        lvs_med = lvs_gb.median()

        # Create corners for leaf voxels
        cnrs = np.vstack([lvs_min.x, lvs_med.y, lvs_med.z]).T
        clstr = np.tile(np.arange(len(lvs_min.index)) + 1 + chull.clstr.max(), 6)
        VX = np.tile(lvs_min.index, 6)
        cnrs = np.vstack([cnrs, np.vstack([lvs_max.x, lvs_med.y, lvs_med.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_min.y, lvs_med.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_max.y, lvs_med.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_med.y, lvs_min.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_med.y, lvs_max.z]).T])

        # Create DataFrame for leaf corners
        leaf_corners = pd.DataFrame(cnrs, columns=['x', 'y', 'z'])
        leaf_corners.loc[:, 'xlabel'] = 1
        leaf_corners.loc[:, 'clstr'] = clstr
        leaf_corners.loc[:, 'VX'] = VX

        # Combine leaves and wood
        branch_and_leaves = leaf_corners.append(chull[['x', 'y', 'z', 'label', 'stem', 'xlabel', 'clstr']])
        branch_and_leaves.reset_index(inplace=True, drop=True)

        # Find neighboring branch and leaf points - used as entry points
        nn = NearestNeighbors(n_neighbors=2).fit(branch_and_leaves[xyz])
        nbl_dist, nbl_idx = nn.kneighbors()
        closest_point_to_leaf = nbl_idx[:len(leaf_corners), :].flatten()    # only leaf points
        closest_branch_points = closest_point_to_leaf[
            np.isin(closest_point_to_leaf, branch_and_leaves.loc[branch_and_leaves.xlabel == 2].index)
        ]

        # Remove all branch points that are not close to leaves
        bal = branch_and_leaves.loc[branch_and_leaves.index.isin(np.unique(
                np.hstack([branch_and_leaves.iloc[:len(leaf_corners)].index.values, closest_branch_points])
              ))]

        # Generate a leaf paths graph
        leaf_paths = generate_path(bal,
                                   bal.loc[bal.xlabel == 2].clstr.unique(),
                                   max_length=1,
                                   n_neighbours=20)
        leaf_paths = leaf_paths.sort_values(['clstr', 'distance'])
        leaf_paths = leaf_paths.loc[~leaf_paths['clstr'].duplicated()]  # remove duplicate paths
        leaf_paths = leaf_paths.loc[leaf_paths.distance > 0]  # remove points within cluster

        # Linking indices to stem number
        top_to_stem = branch_and_leaves.loc[branch_and_leaves.xlabel == 2].set_index('clstr')['stem'].to_dict()
        leaf_paths.loc[:, 't_clstr'] = leaf_paths.t_clstr.map(top_to_stem)

        # Linking indices to VX number
        index_to_VX = branch_and_leaves.loc[branch_and_leaves.xlabel == 1].set_index('clstr')['VX'].to_dict()
        leaf_paths.loc[:, 'VX'] = leaf_paths['clstr'].map(index_to_VX)

        # Same colour for leaves as stem
        lvs = pd.merge(lvs, leaf_paths[['VX', 't_clstr', 'distance']], on='VX', how='leaf')

        # Save leaf voxel data to output
        for leaf in tqdm(in_tile_stem_nodes):
            I = params.base_I[leaf]
            wood_fn = glob.glob(os.path.join(params.odir,
                                             '*' if params.save_diameter_class else '',
                                             f'{params.n:03}_T{I}.leafoff.las'))[0]
            stem = load_file(os.path.join(wood_fn))
            stem.loc[:, 'wood'] = 1

            l_to_a = lvs.loc[lvs.t_clstr == leaf]
            if len(l_to_a) > 0:
                l_to_a.loc[:, 'wood'] = 0
                l_to_a.loc[:, ['red', 'green', 'blue']] = \
                    rgb.loc[rgb.t_clstr == leaf][['red', 'green', 'blue']].values[0] * 1.2
                stem = stem.append(l_to_a[['x', 'y', 'z', 'label',
                                           'red', 'green', 'blue', 't_clstr', 'wood', 'distance']])

            stem = stem.loc[~stem.duplicated()]
            save_file(filename=wood_fn.replace('leafoff', 'leafon'),
                      pointcloud=stem[['x', 'y', 'z', 'red', 'green', 'blue', 'label', 't_clstr', 'wood', 'distance']])

            if params.verbose:
                print(f">> Point cloud with leaves saved to: {wood_fn.replace('leafoff', 'leafon')}.")

    # Finalize and clean up
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    if params.verbose:
        print(f'>> Process finished.')
        print(f'--- Processing completed in {duration.seconds} seconds ---')


if __name__ == "__main__":
    main()