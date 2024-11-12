import time
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


from separation_tools import *
from src.preprocessing import max_points_per_box, pc, pc_global_shift, additional_headers
from fsct.model import Net


batch_size = 10
is_wood_threshold = 1
model_path=os.path.join('./', 'model', 'model.pth')
terrain_class = 0
ground_height_threshold = 0.1
verbose = True

print('\n>> Semantic segmentation started..')
start_time = time.time()

# set device for PyTorch (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if verbose:
    print('-- Using:', device)

# Create PyTorch dataset and dataloader
test_dataset = TestingDataset(root_dir='./',
                              points_per_box=max_points_per_box,
                              device=device)
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0)

# initialize and load the model
model = Net(num_classes=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()  # set model to evaluation mode

# Perform inference on the test dataset
with torch.no_grad():
    output_list = []
    for data in tqdm(test_loader, disable=not verbose):
        data = data.to(device)
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
labels[:, :4] = np.median(classified_pc[indices][:, , -4:], axis=1)  # median of last 4 columns
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
pc = make_dtm(pc, terrain_class=terrain_class)
pc.loc[pc.n_z <= ground_height_threshold, 'label'] = terrain_class

# Save the segmented point cloud
save_file(os.path.join('./', '{}.segmented.{}', format(filename[:-4], output_fmt)),
          pc, additional_headers=['n_z', 'label', 'pTerrain', 'pLeaf', 'pCWD', 'pWood'] + additional_headers)

# Clean up .npy files if not needed
total_time = time.time() - start_time
if not keep_npy:
    [os.unlink(f) for f in test_dataset.filenames]

print(">> Semantic segmentation done in", total_time, "seconds.\n")
