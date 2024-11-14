import time
from tqdm import tqdm

from separation_tools import *


def preprocessing(params):
    if params.verbose:
        print('\n>> Preprocessing Started..')
    start_time = time.time()

    params.point_cloud = os.path.abspath(params.point_cloud)
    params.pc_dir, params.pc_name = os.path.split(params.point_cloud)
    params.base_name = os.path.splitext(params.pc_name)[0]
    params.input_fmt = os.path.splitext(params.point_cloud)[1]

    # create directory structure from params
    params = make_folder_structure(params)

    # read in point cloud
    params.pc, params.additional_headers = load_file(filename=params.point_cloud,
                                                     additional_headers=True,
                                                     verbose=params.verbose)

    # compute plot center, global shift and bounding box
    params.pc_center = compute_plot_centre(params.pc)
    params.global_shift = params.pc[['x', 'y', 'z']].mean()
    params.bbox = compute_bbox(params.pc[['x', 'y', 'z']])

    # downsample the point cloud
    if params.verbose:
        print('\n-- Downsampling to: %s m' % params.subsampling_min_spacing)
    params.pc = downsample(pc=params.pc,
                           v_length=params.subsampling_min_spacing,
                           accurate=False, keep_points=False)

    # apply global shift
    if params.verbose:
        print('\n-- Global shift: ', params.global_shift.values)
    params.pc[['x', 'y', 'z']] = params.pc[['x', 'y', 'z']] - params.global_shift
    params.pc.reset_index(inplace=True)
    params.pc.loc[:, 'pid'] = params.pc.index

    # generate bounding boxes
    xmin, xmax = np.floor(params.pc.x.min()), np.ceil(params.pc.x.max())
    ymin, ymax = np.floor(params.pc.y.min()), np.ceil(params.pc.y.max())
    zmin, zmax = np.floor(params.pc.z.min()), np.ceil(params.pc.z.max())

    box_overlap = params.box_dims[0] * params.box_overlap[0]

    x_cnr = np.arange(xmin - box_overlap, xmax + box_overlap, box_overlap)
    y_cnr = np.arange(ymin - box_overlap, ymax + box_overlap, box_overlap)
    z_cnr = np.arange(zmin - box_overlap, zmax + box_overlap, box_overlap)

    # multithread segmenting points into boxes and update params
    threads = []
    for i, (bx, by, bz) in enumerate(itertools.product(x_cnr, y_cnr, z_cnr)):
        threads.append(threading.Thread(target=save_pts, args=(params, i, bx, by, bz)))

    for x in tqdm(threads,
                  desc='generating data blocks',
                  disable=not params.verbose):
        x.start()

    for x in threads:
        x.join()

    if params.verbose:
        print(f">> Preprocessing Completed in {time.time() - start_time} seconds.\n")

    return params


if __name__ == '__main__':
    # Constants for subsampling and bounding box dimensions
    # -- minimum spacing in metre after downsampling
    subsampling_min_spacing = 0.01
    # -- dimensions of the bounding box
    box_dims = np.array([6, 6, 8])
    # -- overlap for the bounding box
    box_overlap = np.array([0.5, 0.5, 0.25])
    # -- minimun points required in a box to save it
    min_points_per_box = 1000
    # -- maximum points to save in a box
    max_points_per_box = 20000
    # -- directory to save data
    out_path = input('-- Path to save preprocessed tiles: ')

    start_time = time.time()

    pc_file = input('-- Full path to input point cloud: ')
    path_to_point_cloud = os.path.abspath(pc_file)
    pc_dir, pc_name = os.path.split(path_to_point_cloud)
    pc_basename = os.path.splitext(pc_name)[0]
    pc_input_format = os.path.splitext(path_to_point_cloud)[1]

    # read in point cloud
    pc, additional_headers = load_file(filename=path_to_point_cloud,
                                       additional_headers=True,
                                       verbose=True)

    # compute plot centre, global shift and bounding box
    pc_centre = compute_plot_centre(pc)
    pc_global_shift = pc[['x', 'y', 'z']].mean()
    pc_bbox = compute_bbox(pc[['x', 'y', 'z']])

    # downsample the point cloud
    print('>> Downsampling to: %s m' % subsampling_min_spacing)
    pc = downsample(pc=pc,
                    v_length=subsampling_min_spacing,
                    accurate=False,
                    keep_points=False)
    print('>> Downsampling finished.')

    # apply global shift
    print('\n-- Global shift:', pc_global_shift.values)
    pc[['x', 'y', 'z']] = pc[['x', 'y', 'z']] - pc_global_shift
    pc.reset_index(inplace=True)
    pc.loc[:, 'pid'] = pc.index  # assign point IDs

    # generate bounding boxes
    x_min, x_max = np.floor(pc.x.min()), np.ceil(pc.x.max())
    y_min, y_max = np.floor(pc.y.min()), np.ceil(pc.y.max())
    z_min, z_max = np.floor(pc.z.min()), np.ceil(pc.z.max())
    box_overlap = box_dims[0] * box_overlap[0]
    x_cnr = np.arange(x_min - box_overlap, x_max + box_overlap, box_overlap)
    y_cnr = np.arange(y_min - box_overlap, y_max + box_overlap, box_overlap)
    z_cnr = np.arange(z_min - box_overlap, z_max + box_overlap, box_overlap)

    # multithread segmenting points into boxes and save
    threads = []
    for i, (bx, by, bz) in enumerate(itertools.product(x_cnr, y_cnr, z_cnr)):
        threads.append(threading.Thread(target=save_pts,
                                        args=(pc, box_dims, min_points_per_box, max_points_per_box,
                                              i, bx, by, bz, out_path)))
    for x in tqdm(threads, desc='generating data blocks', disable=False):
        x.start()  # start each thread
    for x in threads:
        x.join()  # wait for all threads to finish

    print('>> Preprocessing Finished.')
