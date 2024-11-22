import pickle
from src.supporter.python.separation_tools import *

def identify_stems(params, clusters, stem, bounding_box):
    skeleton = clusters[['x', 'y', 'z', 'n_z', 'n_slice', 'slice']].median().reset_index()
    skeleton.loc[:, 'dbh_node'] = False

    find_stems_min = int(params.find_stems_height // params.slice_thickness)
    find_stems_max = int((params.find_stems_height + params.find_stems_thickness) // params.slice_thickness) + 1
    dbh_nodes_plus = skeleton.loc[skeleton.n_slice.between(find_stems_min, find_stems_max)].clstr
    dbh_slice = stem.loc[stem.clstr.isin(dbh_nodes_plus)]

    if len(dbh_slice) > 0:
        skeleton, dbh_cylinder = find_stems(skl=skeleton, slices=dbh_slice, params=params)

    in_tile_stem_nodes = skeleton.loc[
        skeleton.dbh_node &
        (skeleton.x.between(bounding_box.xmin, bounding_box.xmax)) &
        (skeleton.y.between(bounding_box.ymin, bounding_box.ymax))
        ].clstr

    return skeleton, in_tile_stem_nodes, dbh_cylinder


if __name__ == '__main__':
    find_stems_height = 1.5
    find_stems_thickness = 0.5
    slice_thickness = 0.2
    find_stems_min_radius = 0.025
    find_stems_min_points = 200

    pkl_path = input('-- Full path to input skeleton pickle file: ')
    out_path = input('-- Path to save pickle files of skeleton, stem nodes, and cylinders: ')

    with open(pkl_path, 'rb') as f:
        pkl_dict = pickle.load(f)
    stem_pc = pkl_dict['pc']
    grouped = pkl_dict['clusters']
    bbox = pkl_dict.bounding_box  ## CHECK LINKS

    skeleton = grouped[['x', 'y', 'z', 'n_z', 'n_slice', 'slice']].median().reset_index()
    skeleton.loc[:, 'dbh_node'] = False

    find_stems_min = int(find_stems_height // slice_thickness)
    find_stems_max = int((find_stems_height + find_stems_thickness) // slice_thickness) + 1
    dbh_nodes_plus = skeleton.loc[skeleton.n_slice.between(find_stems_min, find_stems_max)].clstr
    dbh_slice = stem_pc.loc[stem_pc.clstr.isin(dbh_nodes_plus)]

    if len(dbh_slice) > 0:
        skeleton, dbh_cylinder = find_stems(skl=skeleton,
                                            slices=dbh_slice,
                                            find_stems_min_points=find_stems_min_points,
                                            find_stems_min_radius=find_stems_min_radius,
                                            pandarallel=False,
                                            verbose=True)

    in_tile_stem_nodes = skeleton.loc[
        skeleton.dbh_node &
        (skeleton.x.between(bbox.xmin, bbox.xmax)) &
        (skeleton.y.between(bbox.ymin, bbox.ymax))
        ].clstr

    # Generate dictionary to save
    out_dict = {
        'skl': skeleton,
        'nodes': in_tile_stem_nodes,
        'cyl': dbh_cylinder
    }

    # Save dictionary to pickle file
    with open(os.path.join(out_path, 'identify_stems.pkl'), 'wb') as f:
        pickle.dump(out_dict, f)
