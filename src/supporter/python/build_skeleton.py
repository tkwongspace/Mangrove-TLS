import pickle
from src.supporter.python.separation_tools import *

pd.options.mode.chained_assignment = None

def build_skeleton(params):
    # Extract stem points
    stem_pc = params.pc[params.pc.label == 3]

    # Slice stem points
    stem_pc.loc[:, 'slice'] = (stem_pc.z // params.slice_thickness).astype(int) * params.slice_thickness
    stem_pc.loc[:, 'n_slice'] = (stem_pc.n_z // params.slice_thickness).astype(int)

    # Clustering within height slices
    stem_pc.loc[:, 'clstr'] = -1
    stem_pc = slice_cluster(pc=stem_pc, label_offset=0, verbose=params.verbose)

    # Group skeleton points and fit convex hulls
    clusters = stem_pc.loc[stem_pc.clstr != -1].groupby('clstr')
    if params.verbose:
        print('>> Fitting convex hulls to clusters..')
    if params.pandarallel:
        cvx_hull = clusters.parallel_apply(cube).reset_index(drop=True)
    else:
        cvx_hull = clusters.apply(cube, include_groups=False).reset_index()

    return stem_pc, cvx_hull, clusters


if __name__ == '__main__':
    # initial parameters
    slice_thickness = 0.2

    pc_file = input('-- Full path to input point cloud: ')
    out_path = input('-- Path to save pickle files of stems, convex hull and clusters of point clouds: ')

    pc = laspy.read(os.path.abspath(pc_file))
    pc = np.vstack((pc.x, pc.y, pc.z, pc.n_z, pc.label)).T
    pc = pd.DataFrame(data=pc, columns=['x', 'y', 'z', 'n_z', 'label'])

    # Extract stem points
    stem_pc = pc[pc.label == 3]

    # Slice stem points
    stem_pc.loc[:, 'slice'] = (stem_pc.z // slice_thickness).astype(int) * slice_thickness
    stem_pc.loc[:, 'n_slice'] = (stem_pc.n_z // slice_thickness).astype(int)

    # Clustering within height slices
    stem_pc.loc[:, 'clstr'] = -1
    stem_pc = slice_cluster(pc=stem_pc, label_offset=0, verbose=True)

    # Group skeleton points and fit convex hulls
    clusters = stem_pc.loc[stem_pc.clstr != -1].groupby('clstr')
    print('>> Fitting convex hulls to clusters..')
    cvx_hull = clusters.apply(cube).reset_index(drop=True)

    # Generate dictionary to save
    out_dict = {
        'pc': stem_pc,
        'convex_hull': cvx_hull,
        'clusters': clusters
    }

    # Save dictionary to pickle file
    with open(os.path.join(out_path, 'skeleton.pkl'), 'wb') as f:
        pickle.dump(out_dict, f)
