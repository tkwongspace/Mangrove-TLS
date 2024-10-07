%% Main script for performing mangrove woods & leaves segmentation
%
% This script has been tested for point cloud of individual trees,
% but has not yet been tested for point cloud at site scales.
% We suggest to take extra caution when performing segmentation over 
%   point cloud of multiple trees.
%
% This script embeds the following scripts/projects:
% (1) TreeQSM by Inverse group in Tampere University
%       https://github.com/InverseTampere/TreeQSM
% (2) Point cloud regularization developed by Loic Landrieu 
%       https://github.com/loicland/point-cloud-regularization/
% (3) GCMex developed by Veksler, Boykov, Zabih and Kolmogorov 
%       https://github.com/shaibagon/GCMex/tree/master
% (4) LeWos developed by Di Wang 
%       https://github.com/dwang520/LeWoS
%
% Zijian HUANG (c) 2024

%% 1. Point cloud input
path_to_pc = "data/Test_Am8.pcd";
points = pcread(path_to_pc);

% Set parameters for TreeQSM
% -- for detailed explanation of the input structure,
%    please refer to create_input.m in source cold of TreeQSM
% -- parameters are recommended in LeWoS
input_param = struct( ...
    'PatchDiam1', 0.1, ...      % Patch size of the first uniform-size cover
    'PatchDiam2Min', 0.03, ...  % Minimum patch size of the cover sets in the second cover
    'PatchDiam2Max', 0.08, ...  % Maximum cover set size in the stem's base in the second cover
    'lcyl', 3, ...              % Relative (length/radius) length of the cylinders
    'FilRad', 3, ...            % Relative radius for outlier point filtering
    'BallRad1', 0.12, ...       % Ball radius in the first uniform-size cover generation
    'BallRad2', 0.09, ...       % Maximum ball radius in the second cover generation
    'nmin1', 3, ...             % Minimum number of points in BallRad1-balls, generally good value is 3
    'nmin2', 1, ...             % Minimum number of points in BallRad2-balls, generally good value is 1
    'OnlyTree', 1 ...           % If 1, point cloud contains points only from the tree
);

% Set parameters for filtering and downsampling
% -- k-nearest neighbor distance outlier filtering
input_param.filter.k = 10;
% -- distance filter to remove outliers
input_param.filter.nsigma = 1.5;
% -- voxel size (same unit as PC)
input_param.filter.EdgeLength = 0.001;
% -- set radius to 0 to skip filtering by ball neighborhood
input_param.filter.radius = 0.00;
% -- set number of component to 0 to skip small component filtering
input_param.filter.ncomp = 0;
% -- set plot to false to skip plotting after filtering
input_param.filter.plot = false;

%% 2. Filter and downsample the point cloud
points_filtered_idx = filtering(points.Location, input_param);
pc = pointCloud(points.Location(points_filtered_idx, :));

%% 3. Identification of woods and leaves from point cloud
[wood, leaf, label_idx] = MangroveSeg(pc, 0.125, input_param, 1, 1, 1);

