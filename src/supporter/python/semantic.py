import time
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from fsct.model import Net
from torch_geometric.loader import DataLoader
from separation_tools import *


def semantic_segment(params):
    # Check if the coordinates of point cloud are in global coords
    if not np.all(np.isclose(params.pc.loc[['x', 'y', 'z']].mean(), [0, 0, 0], atol=0.1)):
        params.pc[['x', 'y', 'z']] -= params.global_shift

    if params.verbose:
        print('>> Semantic Segmentation Started..')
    params.sem_seg_start_time = time.time()

    # set device for PyTorch (GPU or CPU)
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if params.verbose:
        print('-- Using:', device)

    # Create PyTorch dataset and dataloader
    test_dataset = TestingDataset(root_dir=params.working_dir,
                                  points_per_box=params.max_points_per_box,
                                  device=params.device)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=params.batch_size,
                             shuffle=False,
                             num_workers=0)

    # initialize and load the model
    model = Net(num_classes=4).to(params.device)
    model.load_state_dict(torch.load(params.model, map_location=params.device), strict=False)
    model.eval()  # set model to evaluation mode

    if params.verbose:
        print('>> PyTorch Dataset and Model ready.')
    # Perform inference on the test dataset
    with torch.no_grad():
        output_list = []
        for data in tqdm(test_loader, desc='Performing semantic segmentation', disable=not params.verbose):
            data = data.to(params.device)
            out = model(data)
            out = out.permute(2, 1, 0).squeeze()  # Reshape output
            batches = np.unique(data.batch.cpu())  # Unique batch indices
            out = torch.softmax(out.cpu().detach(), dim=1)  # Apply softmax for probabilities
            pos = data.pos.cpu()
            output = np.hstack((pos, out.numpy()))  # combine positions with probabilities

            # Adjust output for each batch
            for batch in batches:
                output_batch = output[data.batch.cpu() == batch]
                output_batch[:, :3] += data.local_shift.cpu().numpy()[3 * batch:3 + (3 * batch)]
                output_list.append(output_batch)

        classified_pc = np.vstack(output_list)  # stack all output batches

    # clean up anything no longer needed to free RAM.
    del output_batch, out, batches, pos, output

    # choose most confident label for each point
    if params.verbose:
        print(">> Choosing most confident labels..")
    neighbours = NearestNeighbors(n_neighbors=16,
                                  algorithm='kd_tree',
                                  metric='euclidean',
                                  radius=0.05).fit(classified_pc[:, :3])
    _, indices = neighbours.kneighbors(params.pc[['x', 'y', 'z']].values)

    # Drop previous label columns
    params.pc = params.pc.drop(columns=[c for c in params.pc.columns if c in ['label', 'pTerrain', 'pLeaf', 'pWood', 'pCWD']])

    # Create a new labels array based on median probabilities
    labels = np.zeros((params.pc.shape[0], 4))
    labels[:, :4] = np.median(classified_pc[indices][:, :, -4:], axis=1)  # median of last 4 columns
    params.pc.loc[params.pc.index, 'label'] = np.argmax(labels[:, :4], axis=1)  # Assign the most likely label

    # Assign wood class if wood probability exceeds threshold
    # attribute points as wood if any points have a wood probability > is_wood_threshold (Morel et al. 2020)
    is_wood = np.any(classified_pc[indices][:, :, -1] > params.wood_threshold, axis=1)
    params.pc.loc[is_wood, 'label'] = 3  # set label to wood

    probs = pd.DataFrame(index=params.pc.index,
                         data=labels[:, :4],
                         columns=['pTerrain', 'pLeaf', 'pCWD', 'pWood'])
    params.pc = params.pc.join(probs)

    # Adjust point cloud coordinates by global shift
    pc[['x', 'y', 'z']] += pc_global_shift

    # Create DTM and apply ground normalization
    print(">> Making DTM...")
    params.pc = make_dtm(params)
    params.pc.loc[params.pc.n_z <= ground_height_threshold, 'label'] = params.terrain_class

    # Save the segmented point cloud
    save_file(filename=os.path.join(params.out_dir, f'segmented.{params.out_fmt}'),
              pointcloud=params.pc,
              additional_fields=['n_z', 'label', 'pTerrain', 'pLeaf', 'pCWD', 'pWood'] + params.additional_headers)

    # Clean up .npy files if not needed
    params.sem_seg_total_time = time.time() - params.sem_seg_start_time
    if not keep_npy:
        [os.unlink(f) for f in test_dataset.filenames]

    print(">> Semantic segmentation done in", params.sem_seg_total_time, "seconds.\n")

    return params


if __name__ == '__main__':

    batch_size = 10
    is_wood_threshold = 1
    base_path = r"D:\SystemRelated\Documents\GitHub\Mangrove-TLS"
    model_path=os.path.join(base_path, 'src\supporter\python', 'fsct', 'model.pth')
    terrain_class = 0
    ground_height_threshold = 0.1
    max_points_per_box = 20000
    path_to_data_block = os.path.join(base_path, "data\debug_tiles")
    output_fmt = '.las'
    keep_npy = True
    verbose = True

    print('>> Semantic segmentation started..')
    start_time = time.time()

    # read in point cloud
    pc, additional_headers = load_file(filename=os.path.join(base_path, 'data\preprocessed.las'),
                                       additional_headers=True,
                                       verbose=True)
    pc_global_shift = pc[['x', 'y', 'z']].mean()

    # set device for PyTorch (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print('-- Using:', device)

    # Create PyTorch dataset and dataloader
    test_dataset = TestingDataset(root_dir=path_to_data_block,
                                  points_per_box=max_points_per_box,
                                  device=device)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             # drop_last=True,
                             num_workers=0)

    # initialize and load the model
    model = Net(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()  # set model to evaluation mode

    print('>> PyTorch Dataset and Model ready.')
    # Perform inference on the test dataset
    with torch.no_grad():
        output_list = []
        for data in tqdm(test_loader, desc='Performing semantic segmentation'):
            data = data.to(device)
            out = model(data)
            out = out.permute(2, 1, 0).squeeze()  # Reshape output
            batches = np.unique(data.batch.cpu())  # Unique batch indices
            out = torch.softmax(out.cpu().detach(), dim=1)  # Apply softmax for probabilities
            pos = data.pos.cpu()
            # print(f"Position shape: {pos.shape}, Output shape: {out.shape}")  #DEBUG
            output = np.hstack((pos, out.numpy()))  # combine positions with probabilities

            # Adjust output for each batch
            for batch in batches:
                output_batch = output[data.batch.cpu() == batch]
                output_batch[:, :3] += data.local_shift.cpu().numpy()[3 * batch:3 + (3 * batch)]
                # print(f"Output batch shape: {output_batch.shape}")  #DEBUG
                # print(f"Local shift shape: {data.local_shift.shape}, Indices: {3*batch} to {3+3*batch}")  #DEBUG
                output_list.append(output_batch)

        classified_pc = np.vstack(output_list)  # stack all output batches

    # clean up anything no longer needed to free RAM.
    del output_batch, out, batches, pos, output

    # choose most confident label for each point
    if verbose:
        print(">> Choosing most confident labels..")
    neighbours = NearestNeighbors(n_neighbors=16,
                                  algorithm='kd_tree',
                                  metric='euclidean',
                                  radius=0.05).fit(classified_pc[:, :3])
    _, indices = neighbours.kneighbors(pc[['x', 'y', 'z']].values)

    # Drop previous label columns
    pc = pc.drop(columns=[c for c in pc.columns if c in ['label', 'pTerrain', 'pLeaf', 'pWood', 'pCWD']])

    # Create a new labels array based on median probabilities
    labels = np.zeros((pc.shape[0], 4))
    labels[:, :4] = np.median(classified_pc[indices][:, :, -4:], axis=1)  # median of last 4 columns
    pc['label'] = np.argmax(labels[:, :4], axis=1)  # Assign the most likely label

    # Assign wood class if wood probability exceeds threshold
    # attribute points as wood if any points have a wood probability > is_wood_threshold (Morel et al. 2020)
    is_wood = np.any(classified_pc[indices][:, :, -1] > is_wood_threshold, axis=1)
    pc.loc[is_wood, 'label'] = 3  # set label to wood

    probs = pd.DataFrame(index=pc.index,
                         data=labels[:, :4],
                         columns=['pTerrain', 'pLeaf', 'pCWD', 'pWood'])
    pc = pc.join(probs)

    # Adjust point cloud coordinates by global shift
    pc[['x', 'y', 'z']] += pc_global_shift

    # Create DTM and apply ground normalization
    print(">> Making DTM...")
    pc = make_dtm(pc, terrain_class=terrain_class)
    pc.loc[pc.n_z <= ground_height_threshold, 'label'] = terrain_class

    # Save the segmented point cloud
    save_file(filename=os.path.join(base_path, 'data', f'segmented{output_fmt}'),
              pointcloud=pc,
              additional_fields=['n_z', 'label', 'pTerrain', 'pLeaf', 'pCWD', 'pWood'] + additional_headers)

    # Clean up .npy files if not needed
    total_time = time.time() - start_time
    if not keep_npy:
        [os.unlink(f) for f in test_dataset.filenames]

    print(">> Semantic segmentation done in", total_time, "seconds.\n")
