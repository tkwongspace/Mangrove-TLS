import time
from tqdm import tqdm

from separation_tools import *

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

print('\n>> Preprocessing Started..')
start_time = time.time()

pc_file = r''
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
print('\n>>Downsampling to: %s m' % subsampling_min_spacing)
pc = downsample(pc=pc,
                v_length=subsampling_min_spacing,
                accurate=False,
                keep_points=False)

# apply global shift
print('\n-- Global shift: ', pc_global_shift.values)
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
                                          i, bx, by, bz)))
for x in tqdm(threads, desc='generating data blocks', isable=False):
    x.start()  # start each thread
for x in threads:
    x.join() # wait for all threads to finish
