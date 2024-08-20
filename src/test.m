lasReader = lasFileReader('Am7wood.las');
ptCloud = readPointCloud(lasReader);
P = ptCloud.Location;
Pass = filtering(P,inputs);
P = P(Pass,:);
inputs = define_input(P,2,3,2);
%create_input
QSM = treeqsm(P,inputs);

function inputs = define_input(Clouds,nPD1,nPD2Min,nPD2Max)

% ---------------------------------------------------------------------
% DEFINE_INPUT.M       Defines the required inputs (PatchDiam and BallRad 
%                        parameters) for TreeQSM based in estimated tree
%                        radius.
%
% Version 1.0.0
% Latest update     4 May 2022
%
% Copyright (C) 2013-2022 Pasi Raumonen
% ---------------------------------------------------------------------

% Takes in a single tree point clouds, that preferably contains only points 
% from the tree and not e.g. from ground. User defines the number of
% PatchDiam1, PatchDiam2Min, PatchDiam2Max parameter values needed. Then
% the code estimates automatically these parameter values based on the 
% tree stem radius and tree height. Thus this code can be used to generate
% the inputs needed for QSM reconstruction with TreeQSM.
%
% Inputs:
% P         Point cloud of a tree OR string specifying the name of the .mat
%             file where multiple point clouds are saved              
% nPD1      Number of parameter values estimated for PatchDiam1
% nPD2Min   Number of parameter values estimated for PatchDiam2Min
% nPD2Max   Number of parameter values estimated for PatchDiam2Max
%
% Output:
% inputs    Input structure with the estimated parameter values
% ---------------------------------------------------------------------


% Create inputs-structure
create_input
Inputs = inputs;

% If given multiple clouds, extract the names
if ischar(Clouds) || isstring(Clouds)
  matobj = matfile([Clouds,'.mat']);
  names = fieldnames(matobj);
  i = 1;
  n = max(size(names));
  while i <= n && ~strcmp(names{i,:},'Properties')
    i = i+1;
  end
  I = (1:1:n);
  I = setdiff(I,i);
  names = names(I,1);
  names = sort(names);
  nt = max(size(names)); % number of trees/point clouds
else
  P = Clouds;
  nt = 1;
end
inputs(nt).PatchDiam1 = 0;


%% Estimate the PatchDiam and BallRad parameters
for i = 1:nt
  if nt > 1
    % Select point cloud
    P = matobj.(names{i});
    inputs(i) = Inputs;
    inputs(i).name = names{i};
    inputs(i).tree = i;
    inputs(i).plot = 0;
    inputs(i).savetxt = 0;
    inputs(i).savemat = 0;
    inputs(i).disp = 0;
  end

  %% Estimate the stem diameter close to bottom
  % Define height
  Hb = min(P(:,3));
  Ht = max(P(:,3));
  TreeHeight = double(Ht-Hb);
  Hei = P(:,3)-Hb;

  % Select a section (0.02-0.1*tree_height) from the bottom of the tree
  hSecTop = min(4,0.1*TreeHeight);
  hSecBot = 0.02*TreeHeight;
  hSec = hSecTop-hSecBot;
  Sec = Hei > hSecBot & Hei < hSecTop;
  StemBot = P(Sec,1:3);

  % Estimate stem axis (point and direction)
  AxisPoint = mean(StemBot);
  V = StemBot-AxisPoint;
  V = normalize(V);
  AxisDir = optimal_parallel_vector(V);

  % Estimate stem diameter
  d = distances_to_line(StemBot,AxisDir,AxisPoint);
  Rstem = double(median(d));

  % Point resolution (distance between points)
  Res = sqrt((2*pi*Rstem*hSec)/size(StemBot,1));

  %% Define the PatchDiam parameters
  % PatchDiam1 is around stem radius divided by 3.
  pd1 = Rstem/3;%*max(1,TreeHeight/20);
  if nPD1 == 1
    inputs(i).PatchDiam1 = pd1;
  else
    n = nPD1;
    inputs(i).PatchDiam1 = linspace((0.90-(n-2)*0.1)*pd1,(1.10+(n-2)*0.1)*pd1,n);
  end

  % PatchDiam2Min is around stem radius divided by 6 and increased for
  % over 20 m heigh trees.
  pd2 = Rstem/6*min(1,20/TreeHeight);
  if nPD2Min == 1
    inputs(i).PatchDiam2Min = pd2;
  else
    n = nPD2Min;
    inputs(i).PatchDiam2Min = linspace((0.90-(n-2)*0.1)*pd2,(1.10+(n-2)*0.1)*pd2,n);
  end

  % PatchDiam2Max is around stem radius divided by 2.5.
  pd3 = Rstem/2.5;%*max(1,TreeHeight/20);
  if nPD2Max == 1
    inputs(i).PatchDiam2Max = pd3;
  else
    n = nPD2Max;
    inputs(i).PatchDiam2Max = linspace((0.90-(n-2)*0.1)*pd3,(1.10+(n-2)*0.1)*pd3,n);
  end

  % Define the BallRad parameters:
  inputs(i).BallRad1 = max([inputs(i).PatchDiam1+1.5*Res;
    min(1.25*inputs(i).PatchDiam1,inputs(i).PatchDiam1+0.025)]);
  inputs(i).BallRad2 = max([inputs(i).PatchDiam2Max+1.25*Res;
    min(1.2*inputs(i).PatchDiam2Max,inputs(i).PatchDiam2Max+0.025)]);

  %plot_point_cloud(P,1,1)
end
end

function Pass = filtering(P,inputs)

% ---------------------------------------------------------------------
% FILTERING.M       Filters noise from point clouds.
%
% Version 3.0.0
% Latest update     3 May 2022
%
% Copyright (C) 2013-2022 Pasi Raumonen
% ---------------------------------------------------------------------

% Filters the point cloud as follows:
% 
% 1) the possible NaNs are removed.
% 
% 2) (optional, done if filter.k > 0) Statistical kth-nearest neighbor 
% distance outlier filtering based on user defined "k" (filter.k) and
% multiplier for standard deviation (filter.nsigma): Determines the 
% kth-nearest neighbor distance for all points and then removes the points 
% whose distances are over average_distance + nsigma*std. Computes the 
% statistics for each meter layer in vertical direction so that the
% average distances and SDs can change as the point density decreases.
% 
% 3) (optional, done if filter.radius > 0) Statistical point density 
% filtering based on user defined ball radius (filter.radius) and multiplier 
% for standard deviation (filter.nsigma): Balls of radius "filter.radius"
% centered at each point are defined for all points and the number of
% points included ("point density") are computed and then removes the points 
% whose density is smaller than average_density - nsigma*std. Computes the 
% statistics for each meter layer in vertical direction so that the
% average densities and SDs can change as the point density decreases.
% 
% 4) (optional, done if filter.ncomp > 0) Small component filtering based
% on user defined cover (filter.PatchDiam1, filter.BallRad1) and threshold
% (filter.ncomp): Covers the point cloud and determines the connected
% components of the cover and removes the points from the small components
% that have less than filter.ncomp cover sets.
%
% 5) (optional, done if filter.EdgeLength > 0) cubical downsampling of the 
% point cloud based on user defined cube size (filter.EdgeLength): 
% selects randomly one point from each cube
%
% Does the filtering in the above order and thus always applies the next 
% fitering to the point cloud already filtered by the previous methods. 
% Statistical kth-nearest neighbor distance outlier filtering and the 
% statistical point density filtering are meant to be exlusive to each
% other.
%
% Inputs:
% P         Point cloud
% inputs    Inputs structure with the following subfields:
%   filter.EdgeLength   Edge length of the cubes in the cubical downsampling
%   filter.k            k of knn method
%   filter.radius       Radius of the balls in the density filtering
%   filter.nsigma       Multiplier for standard deviation, determines how
%                         far from the mean the threshold is in terms of SD.
%                         Used in both the knn and the density filtering
%   filter.ncomp        Threshold number of components in the small
%                         component filtering
%   filter.PatchDiam1   Defines the patch/cover set size for the component 
%                         filtering
%   filter.BallRad1     Defines the neighbors for the component filtering
%   filter.plot         If true, plots the filtered point cloud
% Outputs:
% Pass      Logical vector indicating points passing the filtering
% ---------------------------------------------------------------------

% Changes from version 2.0.0 to 3.0.0, 3 May 2022:
% Major changes and additions.
% 1) Added two new filtering options: statistical kth-nearest neighbor 
%    distance outlier filtering and cubical downsampling.
% 2) Changed the old point density filtering, which was based on given
%    threshold, into statistical point density filtering, where the
%    threshold is based on user defined statistical measure
% 3) All the input parameters are given by "inputs"-structure that can be
%    defined by "create_input" script   
% 4) Streamlined the coding and what is displayed

%% Initial data processing
% Only double precision data
if ~isa(P,'double')
  P = double(P);
end
% Only x,y,z-data
if size(P,2) > 3
  P = P(:,1:3);
end
np = size(P,1);
np0 = np;
ind = (1:1:np)';
Pass = false(np,1);

disp('----------------------')
disp(' Filtering...')
disp(['  Points before filtering:  ',num2str(np)])

%% Remove possible NaNs
F = ~any(isnan(P),2);
if nnz(F) < np
  disp(['  Points with NaN removed:  ',num2str(np-nnz(Pass))])
  ind = ind(F);
end 

%% Statistical kth-nearest neighbor distance outlier filtering
if inputs.filter.k > 0
  % Compute the knn distances
  Q = P(ind,:);
  np = size(Q,1);
  [~, kNNdist] = knnsearch(Q,Q,'dist','euclidean','k',inputs.filter.k);
  kNNdist = kNNdist(:,end);

  % Change the threshold kNNdistance according the average and standard 
  % deviation for every vertical layer of 1 meter in height
  hmin = min(Q(:,3));
  hmax = max(Q(:,3));
  H = ceil(hmax-hmin);
  F = false(np,1);
  ind = (1:1:np)';
  for i = 1:H
    I = Q(:,3) < hmin+i & Q(:,3) >= hmin+i-1;
    points = ind(I);
    d = kNNdist(points);
    J = d < mean(d)+inputs.filter.nsigma*std(d);
    points = points(J);
    F(points) = 1;
  end
  ind = ind(F);
  disp(['  Points removed as statistical outliers:  ',num2str(np-length(ind))])
end

%% Statistical point density filtering
if inputs.filter.radius > 0
  Q = P(ind,:);
  np = size(Q,1);

  % Partition the point cloud into cubes
  [partition,CC] = cubical_partition(Q,inputs.filter.radius);

  % Determine the number of points inside a ball for each point
  NumOfPoints = zeros(np,1);
  r1 = inputs.filter.radius^2;
  for i = 1:np
    if NumOfPoints(i) == 0
      points = partition(CC(i,1)-1:CC(i,1)+1,CC(i,2)-1:CC(i,2)+1,CC(i,3)-1:CC(i,3)+1);
      points = vertcat(points{:,:});
      cube = Q(points,:);
      p = partition{CC(i,1),CC(i,2),CC(i,3)};
      for j = 1:length(p)
        dist = (Q(p(j),1)-cube(:,1)).^2+(Q(p(j),2)-cube(:,2)).^2+(Q(p(j),3)-cube(:,3)).^2;
        J = dist < r1;
        NumOfPoints(p(j)) = nnz(J);
      end
    end
  end

  % Change the threshold point density according the average and standard 
  % deviation for every vertical layer of 1 meter in height
  hmin = min(Q(:,3));
  hmax = max(Q(:,3));
  H = ceil(hmax-hmin);
  F = false(np,1);
  ind = (1:1:np)';
  for i = 1:H
    I = Q(:,3) < hmin+i & Q(:,3) >= hmin+i-1;
    points = ind(I);
    N = NumOfPoints(points);
    J = N > mean(N)-inputs.filter.nsigma*std(N);
    points = points(J);
    F(points) = 1;
  end
  ind = ind(F);
  disp(['  Points removed as statistical outliers:  ',num2str(np-length(ind))])
end

%% Small component filtering
if inputs.filter.ncomp > 0
  % Cover the point cloud with patches
  input.BallRad1 = inputs.filter.BallRad1;
  input.PatchDiam1 = inputs.filter.PatchDiam1;
  input.nmin1 = 0;
  Q = P(ind,:);
  np = size(Q,1);
  cover = cover_sets(Q,input);

  % Determine the separate components
  Components = connected_components(cover.neighbor,0,inputs.filter.ncomp);

  % The filtering
  B = vertcat(Components{:}); % patches in the components
  points = vertcat(cover.ball{B}); % points in the components
  F = false(np,1);
  F(points) = true;
  ind = ind(F);
  disp(['  Points with small components removed:  ',num2str(np-length(ind))])
end

%% Cubical downsampling
if inputs.filter.EdgeLength > 0
  Q = P(ind,:);
  np = size(Q,1);
  F = cubical_downsampling(Q,inputs.filter.EdgeLength);
  ind = ind(F);
  disp(['  Points removed with downsampling:  ',num2str(np-length(ind))])
end

%% Define the output and display summary results
Pass(ind) = true;
np = nnz(Pass);
disp(['  Points removed in total: ',num2str(np0-np)])
disp(['  Points removed in total (%): ',num2str(round((1-np/np0)*1000)/10)])
disp(['  Points left: ',num2str(np)])

%% Plot the filtered and unfiltered point clouds
if inputs.filter.plot
  plot_comparison(P(Pass,:),P(~Pass,:),1,1,1)
  plot_point_cloud(P(Pass,:),2,1)
end
end

function QSM = treeqsm(P,inputs)

% ---------------------------------------------------------------------
% TREEQSM.M     Reconstructs quantitative structure tree models from point 
%                   clouds containing a tree.
%
% Version 2.4.1
% Latest update     2 May 2022
%
% Copyright (C) 2013-2022 Pasi Raumonen
% ---------------------------------------------------------------------
%
% INPUTS:
%
% P                 (Filtered) point cloud, (m_points x 3)-matrix, the rows
%                       give the coordinates of the points.
%
% inputs            Structure field defining reconstruction parameters.
%                       Created with the "create_input.m" script. Contains 
%                       the following main fields:
%   PatchDiam1        Patch size of the first uniform-size cover
%
%   PatchDiam2Min     Minimum patch size of the cover sets in the second cover
%
%   PatchDiam2Max     Maximum cover set size in the stem's base in the 
%                       second cover
%
%   BallRad1          Ball size used for the first cover generation
%
%   BallRad2          Maximum ball radius used for the second cover generation
%
%   nmin1             Minimum number of points in BallRad1-balls, 
%                       default value is 3.
%
%   nmin2             Minimum number of points in BallRad2-balls, 
%                       default value is 1.
%
%   OnlyTree          If "1", the point cloud contains only points from the 
%                       tree and the trunk's base is defined as the lowest 
%                       part of the point cloud. Default value is "1". 
%
%   Tria              If "1", tries to make triangulation for the stem up 
%                       to first main branch. Default value is "0". 
%
%   Dist              If "1", compute the point-model distances. 
%                       Default value is "1".
%
%   MinCylRad         Minimum cylinder radius, used particularly in the 
%                       taper corrections
%
%   ParentCor         If "1", child branch cylinders radii are always 
%                       smaller than the parent branche's cylinder radii
%
%   TaperCor          If "1", use partially linear (stem) and parabola 
%                       (branches) taper corrections
%
%   GrowthVolCor      If "1", use growth volume correction introduced 
%                       by Jan Hackenberg
%
%   GrowthVolFac      fac-parameter of the growth volume approach, 
%                       defines upper and lower bound
%
%   name              Name string for saving output files and name for the
%                       model in the output object
% 
%   tree              Numerical id/index given to the tree
% 
%   model             Model number of the tree, e.g. with the same inputs
%
%   savemat           If "1", saves the output struct QSM as a matlab-file
%                       into \result folder 
%
%   savetxt           If "1", saves the models in .txt-files into 
%                       \result folder 
%
%   plot              Defines what is plotted during the reconstruction:
%                       2 = same as below plus distributions
%                       1 = plots the segmented point cloud and QSMs
%                       0 = plots nothing
%
%   disp              Defines what is displayed during the reconstruction:
%                       2 = same as below plus times and tree attributes; 
%                       1 = display name, parameters and fit metrics;
%                       0 = display only the name
% ---------------------------------------------------------------------
% OUTPUT:
%
% QSM           Structure array with the following fields:
%               cylinder        Cylinder data  
%               branch          Branch data
%               treedata        Tree attributes  
%               rundata         Information about the modelling run
%               pmdistances     Point-to-model distance statistics
%               triangulation   Triangulation of the stem (if inputs.Tria = 1)
% ---------------------------------------------------------------------

% cylinder (structure-array) contains the following fields:
% radius
% length
% start         xyz-coordinates of the starting point
% axis          xyz-component of the cylinder axis
% parent        index (in this file) of the parent cylinder
% extension     index (in this file) of the extension cylinder
% added         is cylinder added after normal cylinder fitting (= 1 if added)
% UnmodRadius   unmodified radius of the cylinder
% branch        branch (index in the branch structure array) of the cylinder
% BranchOrder   branch order of the branch the cylinder belongs
% PositionInBranch	running number of the cylinder in the branch it belongs
%
% branch (structure-array) contains the following fields:
% order     branch order (0 for trunk, 1 for branches originating from 
%               the trunk, etc.)
% parent	index (in this file) of the parent branch
% volume	volume (L) of the branch (sum of the volumes of the cylinders 
%               forming the branch)
% length	length (m) of the branch (sum of the lengths of the cylinders)
% angle     branching angle (deg) (angle between the branch and its parent 
%               at the branching point)
% height    height (m) of the base of the branch
% azimuth   azimuth (deg) of the branch at the base 
% diameter  diameter (m) of the branch at the base
%
% treedata (structure-array) contains the following fields:
% TotalVolume
% TrunkVolume
% BranchVolume
% TreeHeight
% TrunkLength
% BranchLength
% NumberBranches    Total number of branches
% MaxBranchOrder 
% TotalArea 
% DBHqsm        From the cylinder of the QSM at the right heigth
% DBHcyl        From the cylinder fitted to the section 1.1-1.5m
% location      (x,y,z)-coordinates of the base of the tree
% StemTaper     Stem taper function/curve from the QSM
% VolumeCylDiam     Distribution of the total volume in diameter classes
% LengthCylDiam     Distribution of the total length in diameter classes
% VolumeBranchOrder     Branch volume per branching order
% LengthBranchOrder     Branch length per branching order
% NumberBranchOrder     Number of branches per branching order

% treedata from mixed model (cylinders and triangulation) contains also 
% the following fields:
% DBHtri            Computed from triangulation model
% TriaTrunkVolume   Triangulated trunk volume (up to first branch)
% MixTrunkVolume    Mixed trunk volume, bottom (triang.) + top (cylinders)
% MixTotalVolume    Mixed total volume, mixed trunk volume + branch volume
% TriaTrunkLength   Triangulated trunk length
%
% pmdistances (structure-array) contains the following fields (and others):
% CylDists  Average point-model distance for each cylinder
% median    median of CylDist for all, stem, 1branch, 2branch cylinder
% mean      mean of CylDist for all, stem, 1branch, 2branch cylinder
% max       max of CylDist for all, stem, 1branch, 2branch cylinder
% std       standard dev. of CylDist for all, stem, 1branch, 2branch cylinder
% 
% rundata (structure-array) contains the following fields:
% inputs    The input parameters in a structure-array
% time      Computation times for each step
% date      Starting and stopping dates (year,month,day,hour,minute,second) 
%             of the computation
% 
% triangulation (structure-array) contains the following fields:
% vert      Vertices (xyz-coordinates) of the triangulation
% facet     Facet information
% fvd       Color information for plotting the model
% volume    Volume enclosed by the triangulation
% bottom    Z-coordinate of the bottom plane of the triangulation
% top       Z-coordinate of the top plane of the triangulation
% triah     Height of the triangles
% triah     Width of the triangles
% cylind    Cylinder index in the stem where the triangulation stops
% ---------------------------------------------------------------------

% Changes from version 2.4.0 to 2.4.1, 2 May 2022:  
% Minor update. New filtering options, new code ("define_input") for 
% selecting automatically PatchDiam and BallRad parameter values for 
% the optimization process, added sensitivity estimates of the results, 
% new smoother plotting of QSMs, corrected some bugs, rewrote some 
% functions (e.g. "branches").
% Particular changes in treeqsm.m file:
% 1) Deleted the remove of the field "ChildCyls" and "CylsInSegment".

% Changes from version 2.3.2 to 2.4.0, 17 Aug 2020:  
% First major update. Cylinder fitting process and the taper correction 
% has changed. The fitting is adaptive and no more “lcyl” and “FilRad” 
% parameters. Treedata has many new outputs: Branch and cylinder 
% distributions; surface areas; crown dimensions. More robust triangulation 
% of stem. Branch, cylinder and triangulation structures have new fields. 
% More optimisation metrics, more plots of the results and more plotting 
% functions.
% Particular changes in treeqsm.m file:
% 1) Removed the for-loops for lcyl and FilRad.
% 2) Changes what is displayed about the quality of QSMs 
%    (point-model-distances and surface coverage) during reconstruction
% 3) Added version number to rundata
% 4) Added remove of the field "ChildCyls" and "CylsInSegment" of "cylinder"
%    from "branches" to "treeqsm".

% Changes from version 2.3.1 to 2.3.2, 2 Dec 2019:  
% Small changes in the subfunction to allow trees without branches

% Changes from version 2.3.0 to 2.3.1, 8 Oct 2019:  
% 1) Some changes in the subfunctions, particularly in "cylinders" and 
%    "tree_sets"
% 2) Changed how "treeqsm" displays things during the running of the
%    function


%% Code starts -->
Time = zeros(11,1); % Save computation times for modelling steps
Date = zeros(2,6); % Starting and stopping dates of the computation
Date(1,:) = clock;
% Names of the steps to display
name = ['Cover sets      ';
        'Tree sets       ';
        'Initial segments';
        'Final segments  ';
        'Cylinders       ';
        'Branch & data   ';
        'Distances       '];
 
if inputs.disp > 0
  disp('---------------')
  disp(['  ',inputs.name,', Tree = ',num2str(inputs.tree),...
    ', Model = ',num2str(inputs.model)])
end

% Input parameters
PatchDiam1 = inputs.PatchDiam1;
PatchDiam2Min = inputs.PatchDiam2Min;
PatchDiam2Max = inputs.PatchDiam2Max;
BallRad1 = inputs.BallRad1; 
BallRad2 = inputs.BallRad2; 
nd = length(PatchDiam1);
ni = length(PatchDiam2Min);
na = length(PatchDiam2Max);

if inputs.disp == 2
  % Display parameter values
  disp(['  PatchDiam1 = ',num2str(PatchDiam1)])
  disp(['  BallRad1 = ',num2str(BallRad1)])
  disp(['  PatchDiam2Min = ',num2str(PatchDiam2Min)])
  disp(['  PatchDiam2Max = ',num2str(PatchDiam2Max)])
  disp(['  BallRad2 = ',num2str(BallRad2)])
  disp(['  Tria = ',num2str(inputs.Tria),...
      ', OnlyTree = ',num2str(inputs.OnlyTree)])
  disp('Progress:')
end

%% Make the point cloud into proper form
% only 3-dimensional data
if size(P,2) > 3
    P = P(:,1:3);
end
% Only double precision data
if ~isa(P,'double')
    P = double(P);
end

%% Initialize the output file
QSM = struct('cylinder',{},'branch',{},'treedata',{},'rundata',{},...
    'pmdistance',{},'triangulation',{});

%% Reconstruct QSMs
nmodel = 0;
for h = 1:nd
  tic
  Inputs = inputs;
  Inputs.PatchDiam1 = PatchDiam1(h);
  Inputs.BallRad1 = BallRad1(h);
  if nd > 1 && inputs.disp >= 1
    disp('  -----------------')
    disp(['  PatchDiam1 = ',num2str(PatchDiam1(h))]);
    disp('  -----------------')
  end
  
  %% Generate cover sets
  cover1 = cover_sets(P,Inputs);
  Time(1) = toc;
  if inputs.disp == 2
    display_time(Time(1),Time(1),name(1,:),1)
  end
  
  %% Determine tree sets and update neighbors
  [cover1,Base,Forb] = tree_sets(P,cover1,Inputs);
  Time(2) = toc-Time(1);
  if inputs.disp == 2
    display_time(Time(2),sum(Time(1:2)),name(2,:),1)
  end
  
  %% Determine initial segments
  segment1 = segments(cover1,Base,Forb);
  Time(3) = toc-sum(Time(1:2));
  if inputs.disp == 2
    display_time(Time(3),sum(Time(1:3)),name(3,:),1)
  end
  
  %% Correct segments
  % Don't remove small segments and add the modified base to the segment
  segment1 = correct_segments(P,cover1,segment1,Inputs,0,1,1);
  Time(4) = toc-sum(Time(1:3));
  if inputs.disp == 2
    display_time(Time(4),sum(Time(1:4)),name(4,:),1)
  end
  
  for i = 1:na
    % Modify inputs
    Inputs.PatchDiam2Max = PatchDiam2Max(i);
    Inputs.BallRad2 = BallRad2(i);
    if na > 1 && inputs.disp >= 1
      disp('    -----------------')
      disp(['    PatchDiam2Max = ',num2str(PatchDiam2Max(i))]);
      disp('    -----------------')
    end
    for j = 1:ni
      tic
      % Modify inputs
      Inputs.PatchDiam2Min = PatchDiam2Min(j);
      if ni > 1 && inputs.disp >= 1
        disp('      -----------------')
        disp(['      PatchDiam2Min = ',num2str(PatchDiam2Min(j))]);
        disp('      -----------------')
      end
      
      %% Generate new cover sets
      % Determine relative size of new cover sets and use only tree points
      RS = relative_size(P,cover1,segment1);
      
      % Generate new cover
      cover2 = cover_sets(P,Inputs,RS);
      Time(5) = toc;
      if inputs.disp == 2
          display_time(Time(5),sum(Time(1:5)),name(1,:),1)
      end
      
      %% Determine tree sets and update neighbors
      [cover2,Base,Forb] = tree_sets(P,cover2,Inputs,segment1);
      Time(6) = toc-Time(5);
      if inputs.disp == 2
        display_time(Time(6),sum(Time(1:6)),name(2,:),1)
      end
      
      %% Determine segments
      segment2 = segments(cover2,Base,Forb);
      Time(7) = toc-sum(Time(5:6));
      if inputs.disp == 2
        display_time(Time(7),sum(Time(1:7)),name(3,:),1)
      end
      
      %% Correct segments
      % Remove small segments and the extended bases.
      segment2 = correct_segments(P,cover2,segment2,Inputs,1,1,0);
      Time(8) = toc-sum(Time(5:7));
      if inputs.disp == 2
        display_time(Time(8),sum(Time(1:8)),name(4,:),1)
      end
      
      %% Define cylinders
      cylinder = cylinders(P,cover2,segment2,Inputs);
      Time(9) = toc;
      if inputs.disp == 2
        display_time(Time(9),sum(Time(1:9)),name(5,:),1)
      end
      
      if ~isempty(cylinder.radius)
        %% Determine the branches
        branch = branches(cylinder);
        
        %% Compute (and display) model attributes
        T = segment2.segments{1};
        T = vertcat(T{:});
        T = vertcat(cover2.ball{T});
        trunk = P(T,:); % point cloud of the trunk
        % Compute attributes and distibutions from the cylinder model
        % and possibly some from a triangulation
        [treedata,triangulation] = tree_data(cylinder,branch,trunk,inputs);
        Time(10) = toc-Time(9);
        if inputs.disp == 2
          display_time(Time(10),sum(Time(1:10)),name(6,:),1)
        end
        
        %% Compute point model distances
        if inputs.Dist
          pmdis = point_model_distance(P,cylinder);
          
          % Display the mean point-model distances and surface coverages
          % for stem, branch, 1branc and 2branch cylinders
          if inputs.disp >= 1
            D = [pmdis.TrunkMean pmdis.BranchMean ...
                pmdis.Branch1Mean pmdis.Branch2Mean];
            D = round(10000*D)/10;
            
            T = cylinder.branch == 1;
            B1 = cylinder.BranchOrder == 1;
            B2 = cylinder.BranchOrder == 2;
            SC = 100*cylinder.SurfCov;
            S = [mean(SC(T)) mean(SC(~T)) mean(SC(B1)) mean(SC(B2))];
            S = round(10*S)/10;
            
            disp('  ----------')
            str = ['  PatchDiam1 = ',num2str(PatchDiam1(h)), ...
                ', PatchDiam2Max = ',num2str(PatchDiam2Max(i)), ...
                ', PatchDiam2Min = ',num2str(PatchDiam2Min(j))];
            disp(str)
            str = ['  Distances and surface coverages for ',...
                'trunk, branch, 1branch, 2branch:'];
            disp(str)
            str = ['  Average cylinder-point distance:  '...
                num2str(D(1)),'  ',num2str(D(2)),'  ',...
                num2str(D(3)),'  ',num2str(D(4)),' mm'];
            disp(str)
            str = ['  Average surface coverage:  '...
                num2str(S(1)),'  ',num2str(S(2)),'  ',...
                num2str(S(3)),'  ',num2str(S(4)),' %'];
            disp(str)
            disp('  ----------')
          end
          Time(11) = toc-sum(Time(9:10));
          if inputs.disp == 2
            display_time(Time(11),sum(Time(1:11)),name(7,:),1)
          end
        end
        
        %% Reconstruct the output "QSM"
        Date(2,:) = clock;
        Time(12) = sum(Time(1:11));
        clear qsm
        qsm = struct('cylinder',{},'branch',{},'treedata',{},'rundata',{},...
          'pmdistance',{},'triangulation',{});
        qsm(1).cylinder = cylinder;
        qsm(1).branch = branch;
        qsm(1).treedata = treedata;
        qsm(1).rundata.inputs = Inputs;
        qsm(1).rundata.time = single(Time);
        qsm(1).rundata.date = single(Date);
        qsm(1).rundata.version = '2.4.1';
        if inputs.Dist
          qsm(1).pmdistance = pmdis;
        end
        if inputs.Tria
          qsm(1).triangulation = triangulation;
        end
        nmodel = nmodel+1;
        QSM(nmodel) = qsm;
        
        %% Save the output into results-folder
        % matlab-format (.mat)
        if inputs.savemat
          str = [inputs.name,'_t',num2str(inputs.tree),'_m',...
            num2str(inputs.model)];
          save(['results/QSM_',str],'QSM')
        end
        % text-format (.txt)
        if inputs.savetxt
          if nd > 1 || na > 1 || ni > 1
            str = [inputs.name,'_t',num2str(inputs.tree),'_m',...
              num2str(inputs.model)];
            if nd > 1
              str = [str,'_D',num2str(PatchDiam1(h))];
            end
            if na > 1
              str = [str,'_DA',num2str(PatchDiam2Max(i))];
            end
            if ni > 1
              str = [str,'_DI',num2str(PatchDiam2Min(j))];
            end
          else
            str = [inputs.name,'_t',num2str(inputs.tree),'_m',...
              num2str(inputs.model)];
          end
          save_model_text(qsm,str)
        end

        %% Plot models and segmentations
        if inputs.plot >= 1
          if inputs.Tria
            plot_models_segmentations(P,cover2,segment2,cylinder,trunk,...
                triangulation)
          else
            plot_models_segmentations(P,cover2,segment2,cylinder)
          end
          if nd > 1 || na > 1 || ni > 1
            pause
          end
        end
      end
    end
  end
end
end

function cover = cover_sets(P,inputs,RelSize)

% ---------------------------------------------------------------------
% COVER_SETS.M          Creates cover sets (surface patches) and their
%                       neighbor-relation for a point cloud
%
% Version 2.0.1
% Latest update     2 May 2022
%
% Copyright (C) 2013-2022 Pasi Raumonen
% ---------------------------------------------------------------------

% Covers the point cloud with small sets, which are along the surface,
% such that each point belongs at most one cover set; i.e. the cover is
% a partition of the point cloud.
%
% The cover is generated such that at first the point cloud is covered
% with balls with radius "BallRad". This first cover is such that
% 1) the minimum distance between the centers is "PatchDiam", and
% 2) the maximum distance from any point to nearest center is also "PatchDiam".
% Then the first cover of BallRad-balls is used to define a second cover:
% each BallRad-ball "A" defines corresponding cover set "B" in the second cover
% such that "B" contains those points of "A" that are nearer to the center of
% "A" than any other center of BallRad-balls. The BallRad-balls also define
% the neighbors for the second cover: Let CA and CB denote cover sets in
% the second cover, and BA and BB their BallRad-balls. Then CB is
% a neighbor of CA, and vice versa, if BA and CB intersect or
% BB and CA intersect.
%
% Inputs:
% P         Point cloud
% inputs    Input stucture, the following fields are needed:
%   PatchDiam1   Minimum distance between centers of cover sets; i.e. the
%                   minimum diameter of cover set in uniform covers. Does
%                   not need nor use the third optional input "RelSize".
%   PatchDiam2Min   Minimum diameter of cover sets for variable-size
%                     covers. Needed if "RelSize" is given as input.
%   PatchDiam2Max   Maximum diameter of cover sets for variable-size
%                     covers. Needed if "RelSize" is given as input.
% 	BallRad1    Radius of the balls used to generate the uniform cover. 
%                   These balls are also used to determine the neighbors
%   BallRad2    Maximum radius of the balls used to generate the 
%                   varibale-size cover. 
%   nmin1, nmin2    Minimum number of points in a BallRad1- and
%                       BallRad2-balls
% RelSize   Relative cover set size for each point
%
% Outputs:
% cover     Structure array containing the followin fields:
%   ball        Cover sets, (n_sets x 1)-cell
%   center      Center points of the cover sets, (n_sets x 1)-vector
%   neighbor    Neighboring cover sets of each cover set, (n_sets x 1)-cell

% Changes from version 2.0.0 to 2.0.1, 2 May 2022:
% 1) Added comments and changed some variable names
% 2) Enforced that input parameters are type double

if ~isa(P,'double')
  P = double(P);
end

%% Large balls and centers
np = size(P,1);
Ball = cell(np,1); % Large balls for generation of the cover sets and their neighbors
Cen = zeros(np,1,'uint32'); % the center points of the balls/cover sets
NotExa = true(np,1); % the points not yet examined
Dist = 1e8*ones(np,1);  % distance of point to the closest center
BoP = zeros(np,1,'uint32');  % the balls/cover sets the points belong
nb = 0;             % number of sets generated
if nargin == 2
  %% Same size cover sets everywhere
  BallRad = double(inputs.BallRad1);
  PatchDiamMax = double(inputs.PatchDiam1);
  nmin = double(inputs.nmin1);
  % Partition the point cloud into cubes for quick neighbor search
  [partition,CC] = cubical_partition(P,BallRad);

  % Generate the balls
  Radius = BallRad^2;
  MaxDist = PatchDiamMax^2;
  % random permutation of points, produces different covers for the same inputs:
  RandPerm = randperm(np); 
  for i = 1:np
    if NotExa(RandPerm(i))
      Q = RandPerm(i); % the center/seed point of the current cover set
      % Select the points in the cubical neighborhood of the seed:
      points = partition(CC(Q,1)-1:CC(Q,1)+1,CC(Q,2)-1:CC(Q,2)+1,CC(Q,3)-1:CC(Q,3)+1);
      points = vertcat(points{:});
      % Compute distances of the points to the seed:
      V = [P(points,1)-P(Q,1) P(points,2)-P(Q,2) P(points,3)-P(Q,3)];
      dist = sum(V.*V,2);
      % Select the points inside the ball:
      Inside = dist < Radius;
      if nnz(Inside) >= nmin
        ball = points(Inside); % the points forming the ball
        d = dist(Inside); % the distances of the ball's points
        core = (d < MaxDist); % the core points of the cover set
        NotExa(ball(core)) = false; % mark points as examined
        % define new ball:
        nb = nb+1; 
        Ball{nb} = ball;
        Cen(nb) = Q;
        % Select which points belong to this ball, i.e. are closer this
        % seed than previously tested seeds:
        D = Dist(ball); % the previous distances
        closer = d < D; % which points are closer to this seed
        ball = ball(closer); % define the ball
        % update the ball and distance information of the points
        Dist(ball) = d(closer); 
        BoP(ball) = nb; 
      end
    end
  end
else
  %% Use relative sizes (the size varies)
  % Partition the point cloud into cubes
  BallRad = double(inputs.BallRad2);
  PatchDiamMin = double(inputs.PatchDiam2Min);
  PatchDiamMax = double(inputs.PatchDiam2Max);
  nmin = double(inputs.nmin2);
  MRS = PatchDiamMin/PatchDiamMax;
  % minimum radius
  r = double(1.5*(double(min(RelSize))/256*(1-MRS)+MRS)*BallRad+1e-5); 
  NE = 1+ceil(BallRad/r);
  if NE > 4
    r = PatchDiamMax/4;
    NE = 1+ceil(BallRad/r);
  end
  [Partition,CC,~,Cubes] = cubical_partition(P,r,NE);

  I = RelSize == 0; % Don't use points with no size determined
  NotExa(I) = false;

  % Define random permutation of points (results in different covers for 
  % same input) so that first small sets are generated
  RandPerm = zeros(np,1,'uint32');
  I = RelSize <= 32;
  ind = uint32(1:1:np)';
  I = ind(I);
  t1 = length(I);
  RandPerm(1:1:t1) = I(randperm(t1));
  I = RelSize <= 128 & RelSize > 32;
  I = ind(I);
  t2 = length(I);
  RandPerm(t1+1:1:t1+t2) = I(randperm(t2));
  t2 = t2+t1;
  I = RelSize > 128;
  I = ind(I);
  t3 = length(I);
  RandPerm(t2+1:1:t2+t3) = I(randperm(t3));
  clearvars ind I

  Point = zeros(round(np/1000),1,'uint32');
  e = BallRad-PatchDiamMax;
  for i = 1:np
    if NotExa(RandPerm(i))
      Q = RandPerm(i); % the center/seed point of the current cover set
      % Compute the set size and the cubical neighborhood of the seed point:
      rs = double(RelSize(Q))/256*(1-MRS)+MRS; % relative radius
      MaxDist = PatchDiamMax*rs; % diameter of the cover set
      Radius = MaxDist+sqrt(rs)*e; % radius of the ball including the cover set
      N = ceil(Radius/r); % = number of cells needed to include the ball
      cubes = Cubes(CC(Q,1)-N:CC(Q,1)+N,CC(Q,2)-N:CC(Q,2)+N,CC(Q,3)-N:CC(Q,3)+N);
      I = cubes > 0;
      cubes = cubes(I); % Cubes forming the neighborhood
      Par = Partition(cubes); % cell-array of the points in the neighborhood
      % vertical catenation of the points from the cell-array
      S = cellfun('length',Par);
      stop = cumsum(S);
      start = [0; stop]+1;
      for k = 1:length(stop)
        Point(start(k):stop(k)) = Par{k};
      end
      points = Point(1:stop(k));
      % Compute the distance of the "points" to the seed:
      V = [P(points,1)-P(Q,1) P(points,2)-P(Q,2) P(points,3)-P(Q,3)];
      dist = sum(V.*V,2);
      % Select the points inside the ball:
      Inside = dist < Radius^2;
      if nnz(Inside) >= nmin
        ball = points(Inside); % the points forming the ball
        d = dist(Inside); % the distances of the ball's points
        core = (d < MaxDist^2); % the core points of the cover set
        NotExa(ball(core)) = false; % mark points as examined
        % define new ball:
        nb = nb+1; 
        Ball{nb} = ball;
        Cen(nb) = Q;
        % Select which points belong to this ball, i.e. are closer this
        % seed than previously tested seeds:
        D = Dist(ball); % the previous distances
        closer = d < D; % which points are closer to this seed
        ball = ball(closer); % define the ball
        % update the ball and distance information of the points
        Dist(ball) = d(closer); 
        BoP(ball) = nb; 
      end
    end
  end
end
Ball = Ball(1:nb,:);
Cen = Cen(1:nb);
clearvars RandPerm NotExa Dist

%% Cover sets
% Number of points in each ball and index of each point in its ball
Num = zeros(nb,1,'uint32');
Ind = zeros(np,1,'uint32');
for i = 1:np
  if BoP(i) > 0
    Num(BoP(i)) = Num(BoP(i))+1;
    Ind(i) = Num(BoP(i));
  end
end

% Initialization of the "PointsInSets"
PointsInSets = cell(nb,1);
for i = 1:nb
  PointsInSets{i} = zeros(Num(i),1,'uint32');
end

% Define the "PointsInSets"
for i = 1:np
  if BoP(i) > 0
    PointsInSets{BoP(i),1}(Ind(i)) = i;
  end
end

%% Neighbors
% Define neighbors. Sets A and B are neighbors if the large ball of A
% contains points of B. Notice that this is not a symmetric relation.
Nei = cell(nb,1);
Fal = false(nb,1);
for i = 1:nb
  B = Ball{i};        % the points in the big ball of cover set "i"
  I = (BoP(B) ~= i);
  N = B(I);           % the points of B not in the cover set "i"
  N = BoP(N);

  % select the unique elements of N:
  n = length(N);
  if n > 2
    Include = true(n,1);
    for j = 1:n
      if ~Fal(N(j))
        Fal(N(j)) = true;
      else
        Include(j) = false;
      end
    end
    Fal(N) = false;
    N = N(Include);
  elseif n == 2
    if N(1) == N(2)
      N = N(1);
    end
  end

  Nei{i} = uint32(N);
end

% Make the relation symmetric by adding, if needed, A as B's neighbor
% in the case B is A's neighbor
for i = 1:nb
  N = Nei{i};
  for j = 1:length(N)
    K = (Nei{N(j)} == i);
    if ~any(K)
      Nei{N(j)} = uint32([Nei{N(j)}; i]);
    end
  end
end

% Define output
cover.ball = PointsInSets;
cover.center = Cen;
cover.neighbor = Nei;

%% Display statistics
%disp(['    ',num2str(nb),' cover sets, points not covered: ',num2str(np-nnz(BoP))])
end

function [Partition,CubeCoord,Info,Cubes] = cubical_partition(P,EL,NE)

% ---------------------------------------------------------------------
% CUBICAL_PARTITION.M    Partitions the point cloud into cubes.
%
% Version 1.1.0
% Latest update     6 Oct 2021
%
% Copyright (C) 2015-2021 Pasi Raumonen
% ---------------------------------------------------------------------

% Inputs:
% P           Point cloud, (n_points x 3)-matrix
% EL          Length of the cube edges
% NE          Number of empty edge layers
%
% Outputs:
% Partition   Point cloud partitioned into cubical cells,
%                 (nx x ny x nz)-cell, where nx,ny,nz are the number
%                 of cubes in x,y,z-directions, respectively. If "Cubes"
%                 is outputed, then "Partition" is (n x 1)-cell, where each
%                 cell corresponds to a nonempty cube.
%
% CC          (n_points x 3)-matrix whose rows are the cube coordinates
%                 of each point: x,y,z-coordinates
% Info        The minimum coordinate values and number of cubes in each
%                 coordinate direction
% Cubes       (Optional) (nx x ny x nz)-matrix (array), each nonzero
%                 element indicates that its cube is nonempty and the
%                 number indicates which cell in "Partition" contains the
%                 points of the cube.
% ---------------------------------------------------------------------

% Changes from version 1.0.0 to 1.1.0, 6 Oct 2021:
% 1) Changed the determinationa EL and NE so that the while loop don't
%     continue endlessly in some cases

if nargin == 2
  NE = 3;
end

% The vertices of the big cube containing P
Min = double(min(P));
Max = double(max(P));

% Number of cubes with edge length "EdgeLength" in the sides
% of the big cube
N = double(ceil((Max-Min)/EL)+2*NE+1);
t = 0;
while t < 10 && 8*N(1)*N(2)*N(3) > 4e9
  t = t+1;
  EL = 1.1*EL;
  N = double(ceil((Max-Min)/EL)+2*NE+1);
end
if 8*N(1)*N(2)*N(3) > 4e9
  NE = 3;
  N = double(ceil((Max-Min)/EL)+2*NE+1);
end
Info = [Min N EL NE];

% Calculates the cube-coordinates of the points
CubeCoord = floor([P(:,1)-Min(1) P(:,2)-Min(2) P(:,3)-Min(3)]/EL)+NE+1;

% Sorts the points according a lexicographical order
LexOrd = [CubeCoord(:,1) CubeCoord(:,2)-1 CubeCoord(:,3)-1]*[1 N(1) N(1)*N(2)]';
CubeCoord = uint16(CubeCoord);
[LexOrd,SortOrd] = sort(LexOrd);
SortOrd = uint32(SortOrd);
LexOrd = uint32(LexOrd);

if nargout <= 3
  % Define "Partition"
  Partition = cell(N(1),N(2),N(3));
  np = size(P,1);     % number of points
  p = 1;              % The index of the point under comparison
  while p <= np
    t = 1;
    while (p+t <= np) && (LexOrd(p) == LexOrd(p+t))
      t = t+1;
    end
    q = SortOrd(p);
    Partition{CubeCoord(q,1),CubeCoord(q,2),CubeCoord(q,3)} = SortOrd(p:p+t-1);
    p = p+t;
  end

else
  nc = size(unique(LexOrd),1);

  % Define "Partition"
  Cubes = zeros(N(1),N(2),N(3),'uint32');
  Partition = cell(nc,1);
  np = size(P,1);     % number of points
  p = 1;              % The index of the point under comparison
  c = 0;
  while p <= np
    t = 1;
    while (p+t <= np) && (LexOrd(p) == LexOrd(p+t))
      t = t+1;
    end
    q = SortOrd(p);
    c = c+1;
    Partition{c,1} = SortOrd(p:p+t-1);
    Cubes(CubeCoord(q,1),CubeCoord(q,2),CubeCoord(q,3)) = c;
    p = p+t;
  end
end
end

function display_time(T1,T2,string,display)

% Display the two times given. "T1" is the time named with the "string" and
% "T2" is named "Total".

% Changes 12 Mar 2018: moved the if statement with display from the end to 
%                      the beginning 

if display
    [tmin,tsec] = sec2min(T1);
    [Tmin,Tsec] = sec2min(T2);
    if tmin < 60 && Tmin < 60
        if tmin < 1 && Tmin < 1
            str = [string,' ',num2str(tsec),' sec.   Total: ',num2str(Tsec),' sec'];
        elseif tmin < 1
            str = [string,' ',num2str(tsec),' sec.   Total: ',num2str(Tmin),...
                ' min ',num2str(Tsec),' sec'];
        else
            str = [string,' ',num2str(tmin),' min ',num2str(tsec),...
                ' sec.   Total: ',num2str(Tmin),' min ',num2str(Tsec),' sec'];
        end
    elseif tmin < 60
        Thour = floor(Tmin/60);
        Tmin = Tmin-Thour*60;
        str = [string,' ',num2str(tmin),' min ',num2str(tsec),...
            ' sec.   Total: ',num2str(Thour),' hours ',num2str(Tmin),' min'];
    else
        thour = floor(tmin/60);
        tmin = tmin-thour*60;
        Thour = floor(Tmin/60);
        Tmin = Tmin-Thour*60;
        str = [string,' ',num2str(thour),' hours ',num2str(tmin),...
            ' min.   Total: ',num2str(Thour),' hours ',num2str(Tmin),' min'];
    end
    disp(str)
end
end

function [Tmin,Tsec] = sec2min(T)

% Transforms the given number of seconds into minutes and residual seconds

Tmin = floor(T/60);
Tsec = round((T-Tmin*60)*10)/10;
end

function Set = unique_elements(Set,False)

n = length(Set);
if n > 2
    I = true(n,1);
    for j = 1:n
        if ~False(Set(j))
            False(Set(j)) = true;
        else
            I(j) = false;
        end
    end
    Set = Set(I);
elseif n == 2
    if Set(1) == Set(2)
        Set = Set(1);
    end
end
end

function [cover,Base,Forb] = tree_sets(P,cover,inputs,segment)

% ---------------------------------------------------------------------
% TREE_SETS.M       Determines the base of the trunk and the cover sets
%                   belonging to the tree, updates the neighbor-relation
%
% Version 2.3.0
% Latest update     2 May 2022
%
% Copyright (C) 2013-2022 Pasi Raumonen
% ---------------------------------------------------------------------
%
% Determines the cover sets that belong to the tree. Determines also the
% base of the tree and updates the neighbor-relation such that all of the
% tree is connected, i.e., the cover sets belonging to the tree form a
% single connected component. Optionally uses information from existing
% segmentation to make sure that stem and 1st-, 2nd-, 3rd-order branches
% are properly connnected.
% ---------------------------------------------------------------------
% Inputs:
% P             Point cloud
% cover         Cover sets, their centers and neighbors
% PatchDiam     Minimum diameter of the cover sets
% OnlyTree      Logical value indicating if the point cloud contains only
%                   points from the tree to be modelled
% segment       Previous segments
%
% Outputs:
% cover     Cover sets with updated neigbors
% Base      Base of the trunk (the cover sets forming the base)
% Forb      Cover sets not part of the tree
% ---------------------------------------------------------------------

% Changes from version 2.2.0 to 2.3.0, 2 May 2022:
% 1) Added new lines of code at the end of the "define_main_branches" to
%    make sure that the "Trunk" variable defines connected stem

% Changes from version 2.1.0 to 2.2.0, 13 Aug 2020:
% 1) "define_base_forb": Changed the base height specification from
%     0.1*aux.Height to 0.02*aux.Height
% 2) "define_base_forb": changed the cylinder fitting syntax corresponding
%     to the new input and outputs of "least_squares_cylinder"
% 3) "make_tree_connected”: Removed "Trunk(Base) = false;" at the beginning
%     of the function as unnecessary and to prevent errors in a special case
%     where the Trunk is equal to Base.
%	4) "make_tree_connected”: Removed from the end the generation of "Trunk"
%     again and the new call for the function
%	5) "make_tree_connected”: Increased the minimum distance of a component
%     to be removed from 8m to 12m.

% Changes from version 2.0.0 to 2.1.0, 11 Oct 2019:
% 1) "define_main_branches": modified the size of neighborhood "balls0",
%    added seven lines of code, prevents possible error of too low or big
%    indexes on "Par"
% 2) Increased the maximum base height from 0.5m to 1.5m
% 3) "make_tree_connected": added at the end a call for the function itself,
%    if the tree is not yet connected, thus running the function again if
%    necessary

%% Define auxiliar object
clear aux
aux.nb = max(size(cover.center));   % number of cover sets
aux.Fal = false(aux.nb,1);
aux.Ind = (1:1:aux.nb)';
aux.Ce = P(cover.center,1:3); % Coordinates of the center points
aux.Hmin = min(aux.Ce(:,3));
aux.Height = max(aux.Ce(:,3))-aux.Hmin;

%% Define the base of the trunk and the forbidden sets
if nargin == 3
  [Base,Forb,cover] = define_base_forb(P,cover,aux,inputs);
else
  inputs.OnlyTree = true;
  [Base,Forb,cover] = define_base_forb(P,cover,aux,inputs,segment);
end

%% Define the trunk (and the main branches)
if nargin == 3
  [Trunk,cover] = define_trunk(cover,aux,Base,Forb,inputs);
else
  [Trunk,cover] = define_main_branches(cover,segment,aux,inputs);
end

%% Update neighbor-relation to make the whole tree connected
[cover,Forb] = make_tree_connected(cover,aux,Forb,Base,Trunk,inputs);

end % End of the main function


function [Base,Forb,cover] = define_base_forb(P,cover,aux,inputs,segment)

% Defines the base of the stem and the forbidden sets (the sets containing
% points not from the tree, i.e, ground, understory, etc.)
Ce = aux.Ce;
if inputs.OnlyTree && nargin == 4
  % No ground in the point cloud, the base is the lowest part
  BaseHeight = min(1.5,0.02*aux.Height);
  I = Ce(:,3) < aux.Hmin+BaseHeight;
  Base = aux.Ind(I);
  Forb = aux.Fal;
  % Make sure the base, as the bottom of point cloud, is not in multiple parts
  Wb = max(max(Ce(Base,1:2))-min(Ce(Base,1:2)));
  Wt = max(max(Ce(:,1:2))-min(Ce(:,1:2)));
  k = 1;
  while k <= 5 && Wb > 0.3*Wt
    BaseHeight = BaseHeight-0.05;
    BaseHeight = max(BaseHeight,0.05);
    if BaseHeight > 0
      I = Ce(:,3) < aux.Hmin+BaseHeight;
    else
      [~,I] = min(Ce(:,3));
    end
    Base = aux.Ind(I);
    Wb = max(max(Ce(Base,1:2))-min(Ce(Base,1:2)));
    k = k+1;
  end
elseif inputs.OnlyTree
  % Select the stem sets from the previous segmentation and define the
  % base
  BaseHeight = min(1.5,0.02*aux.Height);
  SoP = segment.SegmentOfPoint(cover.center);
  stem = aux.Ind(SoP == 1);
  I = Ce(stem,3) < aux.Hmin+BaseHeight;
  Base = stem(I);
  Forb = aux.Fal;
else
  % Point cloud contains non-tree points.
  % Determine the base from the "height" and "density" of cover sets
  % by projecting the sets to the xy-plane
  Bal = cover.ball;
  Nei = cover.neighbor;

  % The vertices of the rectangle containing C
  Min = double(min(Ce));
  Max = double(max(Ce(:,1:2)));

  % Number of rectangles with edge length "E" in the plane
  E = min(0.1,0.015*aux.Height);
  n = double(ceil((Max(1:2)-Min(1:2))/E)+1);

  % Calculates the rectangular-coordinates of the points
  px = floor((Ce(:,1)-Min(1))/E)+1;
  py = floor((Ce(:,2)-Min(2))/E)+1;

  % Sorts the points according a lexicographical order
  LexOrd = [px py-1]*[1 n(1)]';
  [LexOrd,SortOrd] = sort(LexOrd);

  Partition = cell(n(1),n(2));
  hei = zeros(n(1),n(2)); % "height" of the cover sets in the squares
  den = hei;  % density of the cover sets in the squares
  baseden = hei;
  p = 1; % The index of the point under comparison
  while p <= aux.nb
    t = 1;
    while (p+t <= aux.nb) && (LexOrd(p) == LexOrd(p+t))
      t = t+1;
    end
    q = SortOrd(p);
    J = SortOrd(p:p+t-1);
    Partition{px(q),py(q)} = J;
    p = p+t;
    K = ceil(10*(Ce(J,3)-Min(3)+0.01)/(aux.Height-0.01));
    B = K <= 2;
    K = unique(K);
    hei(px(q),py(q)) = length(K)/10;
    den(px(q),py(q)) = t;
    baseden(px(q),py(q)) = nnz(B);
  end
  den = den/max(max(den));  % normalize
  baseden = baseden/max(max(baseden));

  % function whose maximum determines location of the trunk
  f = den.*hei.*baseden;
  % smooth the function by averaging over 8-neighbors
  x = zeros(n(1),n(2));
  y = zeros(n(1),n(2));
  for i = 2:n(1)-1
    for j = 2:n(2)-1
      f(i,j) = mean(mean(f(i-1:i+1,j-1:j+1)));
      x(i,j) = Min(1)+i*E;
      y(i,j) = Min(2)+j*E;
    end
  end
  f = f/max(max(f));

  % Trunk location is around the maximum f-value
  I = f > 0.5;
  Trunk0 = Partition(I); % squares that contain the trunk
  Trunk0 = vertcat(Trunk0{:});
  HBottom = min(Ce(Trunk0,3));
  I = Ce(Trunk0,3) > HBottom+min(0.02*aux.Height,0.3);
  J = Ce(Trunk0,3) < HBottom+min(0.08*aux.Height,1.5);
  I = I&J; % slice close to bottom should contain the trunk
  Trunk = Trunk0(I);
  Trunk = union(Trunk,vertcat(Nei{Trunk})); % Expand with neighbors
  Trunk = union(Trunk,vertcat(Nei{Trunk})); % Expand with neighbors
  Trunk = union(Trunk,vertcat(Nei{Trunk})); % Expand with neighbors

  % Define connected components of Trunk and select the largest component
  [Comp,CS] = connected_components(Nei,Trunk,0,aux.Fal);
  [~,I] = max(CS);
  Trunk = Comp{I};

  % Fit cylinder to Trunk
  I = Ce(Trunk,3) < HBottom+min(0.1*aux.Height,2); % Select the bottom part
  Trunk = Trunk(I);
  Trunk = union(Trunk,vertcat(Nei{Trunk}));
  Points = Ce(Trunk,:);
  c.start = mean(Points);
  c.axis = [0 0 1];
  c.radius = mean(distances_to_line(Points,c.axis,c.start));
  c = least_squares_cylinder(Points,c);

  % Remove far away points and fit new cylinder
  dis = distances_to_line(Points,c.axis,c.start);
  [~,I] = sort(abs(dis));
  I = I(1:ceil(0.9*length(I)));
  Points = Points(I,:);
  Trunk = Trunk(I);
  c = least_squares_cylinder(Points,c);

  % Select the sets in the bottom part of the trunk and remove sets too
  % far away form the cylinder axis (also remove far away points from sets)
  I = Ce(Trunk0,3) < HBottom+min(0.04*aux.Height,0.6);
  TrunkBot = Trunk0(I);
  TrunkBot = union(TrunkBot,vertcat(Nei{TrunkBot}));
  TrunkBot = union(TrunkBot,vertcat(Nei{TrunkBot}));
  n = length(TrunkBot);
  Keep = true(n,1); % Keep sets that are close enough the axis
  a = max(0.06,0.2*c.radius);
  b = max(0.04,0.15*c.radius);
  for i = 1:n
    d = distances_to_line(Ce(TrunkBot(i),:),c.axis,c.start);
    if d < c.radius+a
      B = Bal{Trunk(i)};
      d = distances_to_line(P(B,:),c.axis,c.start);
      I = d < c.radius+b;
      Bal{Trunk(i)} = B(I);
    else
      Keep(i) = false;
    end
  end
  TrunkBot = TrunkBot(Keep);

  % Select the above part of the trunk and combine with the bottom
  I = Ce(Trunk0,3) > HBottom+min(0.03*aux.Height,0.45);
  Trunk = Trunk0(I);
  Trunk = union(Trunk,vertcat(Nei{Trunk}));
  Trunk = union(Trunk,TrunkBot);

  BaseHeight = min(1.5,0.02*aux.Height);
  % Determine the base
  Bot = min(Ce(Trunk,3));
  J = Ce(Trunk,3) < Bot+BaseHeight;
  Base = Trunk(J);

  % Determine "Forb", i.e, ground and non-tree sets by expanding Trunk
  % as much as possible
  Trunk = union(Trunk,vertcat(Nei{Trunk}));
  Forb = aux.Fal;
  Ground = setdiff(vertcat(Nei{Base}),Trunk);
  Ground = setdiff(union(Ground,vertcat(Nei{Ground})),Trunk);
  Forb(Ground) = true;
  Forb(Base) = false;
  Add = Forb;
  while any(Add)
    Add(vertcat(Nei{Add})) = true;
    Add(Forb) = false;
    Add(Trunk) = false;
    Forb(Add) = true;
  end

  % Try to expand the "Forb" more by adding all the bottom sets
  Bot = min(Ce(Trunk,3));
  Ground = Ce(:,3) < Bot+0.03*aux.Height;
  Forb(Ground) = true;
  Forb(Trunk) = false;
  cover.ball = Bal;
end

end % End of function


function [Trunk,cover] = define_trunk(cover,aux,Base,Forb,inputs)

% This function tries to make sure that likely "route" of the trunk from
% the bottom to the top is connected. However, this does not mean that the
% final trunk follows this "route".

Nei = cover.neighbor;
Ce = aux.Ce;
% Determine the output "Trunk" which indicates which sets are part of
% likely trunk
Trunk = aux.Fal;
Trunk(Base) = true;
% Expand Trunk from the base above with neighbors as long as possible
Exp = Base; % the current "top" of Trunk
% select the unique neighbors of Exp
Exp = unique_elements([Exp; vertcat(Nei{Exp})],aux.Fal);
I = Trunk(Exp);
J = Forb(Exp);
Exp = Exp(~I|~J); % Only non forbidden sets that are not already in Trunk
Trunk(Exp) = true; % Add the expansion Exp to Trunk
L = 0.25; % maximum height difference in Exp from its top to bottom
H = max(Ce(Trunk,3))-L; % the minimum bottom heigth for the current Exp
% true as long as the expansion is possible with original neighbors:
FirstMod = true;
while ~isempty(Exp)
  % Expand Trunk similarly as above as long as possible
  H0 = H;
  Exp0 = Exp;
  Exp = union(Exp,vertcat(Nei{Exp}));
  I = Trunk(Exp);
  Exp = Exp(~I);
  I = Ce(Exp,3) >= H;
  Exp = Exp(I);
  Trunk(Exp) = true;
  if ~isempty(Exp)
    H = max(Ce(Exp,3))-L;
  end

  % If the expansion Exp is empty and the top of the tree is still over 5
  % meters higher, then search new neighbors from above
  if (isempty(Exp) || H < H0+inputs.PatchDiam1/2) && H < aux.Height-5

    % Generate rectangular partition of the sets
    if FirstMod
      FirstMod = false;
      % The vertices of the rectangle containing C
      Min = double(min(Ce(:,1:2)));
      Max = double(max(Ce(:,1:2)));
      nb = size(Ce,1);

      % Number of rectangles with edge length "E" in the plane
      EdgeLenth = 0.2;
      NRect = double(ceil((Max-Min)/EdgeLenth)+1);

      % Calculates the rectangular-coordinates of the points
      px = floor((Ce(:,1)-Min(1))/EdgeLenth)+1;
      py = floor((Ce(:,2)-Min(2))/EdgeLenth)+1;

      % Sorts the points according a lexicographical order
      LexOrd = [px py-1]*[1 NRect(1)]';
      [LexOrd,SortOrd] = sort(LexOrd);

      Partition = cell(NRect(1),NRect(2));
      p = 1; % The index of the point under comparison
      while p <= nb
        t = 1;
        while (p+t <= nb) && (LexOrd(p) == LexOrd(p+t))
          t = t+1;
        end
        q = SortOrd(p);
        J = SortOrd(p:p+t-1);
        Partition{px(q),py(q)} = J;
        p = p+t;
      end
    end

    % Select the region that is connected to a set above it
    if ~isempty(Exp)
      Region = Exp;
    else
      Region = Exp0;
    end

    % Select the minimum and maximum rectangular coordinate of the
    % region
    X1 = min(px(Region));
    if X1 <= 2
      X1 = 3;
    end
    X2 = max(px(Region));
    if X2 >= NRect(1)-1
      X2 = NRect(1)-2;
    end
    Y1 = min(py(Region));
    if Y1 <= 2
      Y1 = 3;
    end
    Y2 = max(py(Region));
    if Y2 >= NRect(2)-1
      Y2 = NRect(2)-2;
    end

    % Select the sets in the 2 meter layer above the region
    sets = Partition(X1-2:X2+2,Y1-2:Y2+2);
    sets = vertcat(sets{:});
    K = aux.Fal;
    K(sets) = true; % the potential sets
    I = Ce(:,3) > H;
    J = Ce(:,3) < H+2;
    I = I&J&K;
    I(Trunk) = false; % Must be non-Trunk sets
    SetsAbove = aux.Ind(I);

    % Search the closest connection between Region and SetsAbove that
    % is enough upward sloping (angle to the vertical has cosine larger
    % than 0.7)
    if ~isempty(SetsAbove)
      % Compute the distances and cosines of the connections
      n = length(Region);
      m = length(SetsAbove);
      Dist = zeros(n,m);
      Cos = zeros(n,m);
      for i = 1:n
        V = mat_vec_subtraction(Ce(SetsAbove,:),Ce(Region(i),:));
        Len = sum(V.*V,2);
        v = normalize(V);
        Dist(i,:) = Len';
        Cos(i,:) = v(:,3)';
      end
      I = Cos > 0.7; % select those connection with large enough cosines
      % if not any, search with smaller cosines
      t = 0;
      while ~any(I)
        t = t+1;
        I = Cos > 0.7-t*0.05;
      end
      % Search the minimum distance
      Dist(~I) = 3;
      if n > 1 && m > 1
        [d,I] = min(Dist);
        [~,J] = min(d);
        I = I(J);
      elseif n == 1 && m > 1
        [~,J] = min(Dist);
        I = 1;
      elseif m == 1 && n < 1
        [~,I] = min(Dist);
        J = 1;
      else
        I = 1; % the set in component to be connected
        J = 1; % the set in "trunk" to be connected
      end

      % Join to "SetsAbove"
      I = Region(I);
      J = SetsAbove(J);
      % make the connection
      Nei{I} = [Nei{I}; J];
      Nei{J} = [Nei{J}; I];

      % Expand "Trunk" again
      Exp = union(Region,vertcat(Nei{Region}));
      I = Trunk(Exp);
      Exp = Exp(~I);
      I = Ce(Exp,3) >= H;
      Exp = Exp(I);
      Trunk(Exp) = true;
      H = max(Ce(Exp,3))-L;
    end
  end
end
cover.neighbor = Nei;

end % End of function


function [Trunk,cover] = define_main_branches(cover,segment,aux,inputs)

% If previous segmentation exists, then use it to make the sets in its main
% branches (stem and first (second or even up to third) order branches)
% connected. This ensures that similar branching structure as in the
% existing segmentation is possible.

Bal = cover.ball;
Nei = cover.neighbor;
Ce = aux.Ce;
% Determine sets in the main branches of previous segmentation
nb = size(Bal,1);
MainBranches = zeros(nb,1);
SegmentOfPoint = segment.SegmentOfPoint;
% Determine which branch indexes define the main branches
MainBranchIndexes = false(max(SegmentOfPoint),1);
MainBranchIndexes(1) = true;
MainBranchIndexes(segment.branch1indexes) = true;
MainBranchIndexes(segment.branch2indexes) = true;
MainBranchIndexes(segment.branch3indexes) = true;
for i = 1:nb
  BranchInd = nonzeros(SegmentOfPoint(Bal{i}));
  if ~isempty(BranchInd)
    ind = min(BranchInd);
    if MainBranchIndexes(ind)
      MainBranches(i) = min(BranchInd);
    end
  end
end

% Define the trunk sets
Trunk = aux.Fal;
Trunk(MainBranches > 0) = true;

% Update the neighbors to make the main branches connected
[Par,CC] = cubical_partition(Ce,3*inputs.PatchDiam2Max,10);
Sets = zeros(aux.nb,1,'uint32');
BI = max(MainBranches);
N = size(Par);
for i = 1:BI
  if MainBranchIndexes(i)
    Branch = MainBranches == i; % The sets forming branch "i"
    % the connected components of "Branch":
    Comps = connected_components(Nei,Branch,1,aux.Fal);
    n = size(Comps,1);
    % Connect the components to each other as long as there are more than
    % one component
    while n > 1
      for j = 1:n
        comp = Comps{j};
        NC = length(comp);

        % Determine branch sets closest to the component
        c = unique(CC(comp,:),'rows');
        m = size(c,1);
        t = 0;
        NearSets = zeros(0,1);
        while isempty(NearSets)
          NearSets = aux.Fal;
          t = t+1;
          for k = 1:m
            x1 = max(1,c(k,1)-t);
            x2 = min(c(k,1)+t,N(1));
            y1 = max(1,c(k,2)-t);
            y2 = min(c(k,2)+t,N(2));
            z1 = max(1,c(k,3)-t);
            z2 = min(c(k,3)+t,N(3));
            balls0 = Par(x1:x2,y1:y2,z1:z2);
            if t == 1
              balls = vertcat(balls0{:});
            else
              S = cellfun('length',balls0);
              I = S > 0;
              S = S(I);
              balls0 = balls0(I);
              stop = cumsum(S);
              start = [0; stop]+1;
              for l = 1:length(stop)
                Sets(start(l):stop(l)) = balls0{l};
              end
              balls = Sets(1:stop(l));
            end
            I = Branch(balls);
            balls = balls(I);
            NearSets(balls) = true;
          end
          NearSets(comp) = false; % Only the non-component cover sets
          NearSets = aux.Ind(NearSets);
        end

        % Determine the closest sets for "comp"
        if ~isempty(NearSets)
          d = pdist2(Ce(comp,:),Ce(NearSets,:));
          if NC == 1 && length(NearSets) == 1
            IU = 1; % the set in component to be connected
            JU = 1; % the set in "trunk" to be connected
          elseif NC == 1
            [du,JU] = min(d);
            IU = 1;
          elseif length(NearSets) == 1
            [du,IU] = min(d);
            JU = 1;
          else
            [d,IU] = min(d);
            [du,JU] = min(d);
            IU = IU(JU);
          end

          % Join to the closest component
          I = comp(IU);
          J = NearSets(JU);
          % make the connection
          Nei{I} = [Nei{I}; J];
          Nei{J} = [Nei{J}; I];
        end
      end

      Comps = connected_components(Nei,Branch,1,aux.Fal);
      n = size(Comps,1);
    end
  end
end

% Update the neigbors to connect 1st-order branches to the stem
Stem = MainBranches == 1;
Stem = aux.Ind(Stem);
MainBranchIndexes = false(max(SegmentOfPoint),1);
MainBranchIndexes(segment.branch1indexes) = true;
BI = max(segment.branch1indexes);
if isempty(BI)
  BI = 0;
end
for i = 2:BI
  if MainBranchIndexes(i)
    Branch = MainBranches == i;
    Branch = aux.Ind(Branch);
    if ~isempty(Branch)
      Neigbors = MainBranches(vertcat(Nei{Branch})) == 1;
      if ~any(Neigbors)
        d = pdist2(Ce(Branch,:),Ce(Stem,:));
        if length(Branch) > 1 && length(Stem) > 1
          [d,I] = min(d);
          [d,J] = min(d);
          I = I(J);
        elseif length(Branch) == 1 && length(Stem) > 1
          [d,J] = min(d);
          I = 1;
        elseif length(Stem) == 1 && length(Branch) > 1
          [d,I] = min(d);
          J = 1;
        elseif length(Branch) == 1 && length(Stem) == 1
          I = 1; % the set in component to be connected
          J = 1; % the set in "trunk" to be connected
        end

        % Join the Branch to Stem
        I = Branch(I);
        J = Stem(J);
        Nei{I} = [Nei{I}; J];
        Nei{J} = [Nei{J}; I];
      end
    end
  end
end
cover.neighbor = Nei;

% Check if the trunk is still in mutliple components and select the bottom
% component to define "Trunk":
[comps,cs] = connected_components(cover.neighbor,Trunk,aux.Fal);
if length(cs) > 1
  [cs,I] = sort(cs,'descend');
  comps = comps(I);
  Stem = MainBranches == 1;
  Trunk = aux.Fal;
  i = 1;
  C = comps{i};
  while i <= length(cs) && ~any(Stem(C))
    i = i+1;
    C = comps{i};
  end
  Trunk(C) = true;
end


end % End of function


function [cover,Forb] = make_tree_connected(cover,aux,Forb,Base,Trunk,inputs)

% Update neighbor-relation for whole tree such that the whole tree is one
% connected component

Nei = cover.neighbor;
Ce = aux.Ce;
% Expand trunk as much as possible
Trunk(Forb) = false;
Exp = Trunk;
while any(Exp)
  Exp(vertcat(Nei{Exp})) = true;
  Exp(Trunk) = false;
  Exp(Forb) = false;
  Exp(Base) = false;
  Trunk(Exp) = true;
end

% Define "Other", sets not yet connected to trunk or Forb
Other = ~aux.Fal;
Other(Forb) = false;
Other(Trunk) = false;
Other(Base) = false;

% Determine parameters on the extent of the "Nearby Space" and acceptable
% component size
% cell size for "Nearby Space" = k0 times PatchDiam:
k0 = min(10,ceil(0.2/inputs.PatchDiam1));
% current cell size, increases by k0 every time when new connections cannot
% be made:
k = k0;
if inputs.OnlyTree
  Cmin = 0;
else
  Cmin = ceil(0.1/inputs.PatchDiam1);  % minimum accepted component size,
  % smaller ones are added to Forb, the size triples every round
end

% Determine the components of "Other"
if any(Other)
  Comps = connected_components(Nei,Other,1,aux.Fal);
  nc = size(Comps,1);
  NonClassified = true(nc,1);
  %plot_segs(P,Comps,6,1,cover.ball)
  %pause
else
  NonClassified = false;
end

bottom = min(Ce(Base,3));
% repeat search and connecting as long as "Other" sets exists
while any(NonClassified)
  npre = nnz(NonClassified); % number of "Other" sets before new connections
  again = true; % check connections again with same "distance" if true

  % Partition the centers of the cover sets into cubes with size k*dmin
  [Par,CC] = cubical_partition(Ce,k*inputs.PatchDiam1);
  Neighbors = cell(nc,1);
  Sizes = zeros(nc,2);
  Pass = true(nc,1);
  first_round = true;
  while again
    % Check each component: part of "Tree" or "Forb"
    for i = 1:nc
      if NonClassified(i) && Pass(i)
        comp = Comps{i}; % candidate component for joining to the tree

        % If the component is neighbor of forbidden sets, remove it
        J = Forb(vertcat(Nei{comp}));
        if any(J)
          NonClassified(i) = false;
          Forb(comp) = true;
          Other(comp) = false;
        else
          % Other wise check nearest sets for a connection
          NC = length(comp);
          if first_round

            % Select the cover sets the nearest to the component
            c = unique(CC(comp,:),'rows');
            m = size(c,1);
            B = cell(m,1);
            for j = 1:m
              balls = Par(c(j,1)-1:c(j,1)+1,...
                c(j,2)-1:c(j,2)+1,c(j,3)-1:c(j,3)+1);
              B{j} = vertcat(balls{:});
            end
            NearSets = vertcat(B{:});
            % Only the non-component cover sets
            aux.Fal(comp) = true;
            I = aux.Fal(NearSets);
            NearSets = NearSets(~I);
            aux.Fal(comp) = false;
            NearSets = unique(NearSets);
            Neighbors{i} = NearSets;
            if isempty(NearSets)
              Pass(i) = false;
            end
            % No "Other" sets
            I = Other(NearSets);
            NearSets = NearSets(~I);
          else
            NearSets = Neighbors{i};
            % No "Other" sets
            I = Other(NearSets);
            NearSets = NearSets(~I);
          end

          % Select different class from NearSets
          I = Trunk(NearSets);
          J = Forb(NearSets);
          trunk = NearSets(I); % "Trunk" sets
          forb = NearSets(J); % "Forb" sets
          if length(trunk) ~= Sizes(i,1) || length(forb) ~= Sizes(i,2)
            Sizes(i,:) = [length(trunk) length(forb)];

            % If large component is tall and close to ground, then
            % search the connection near the component's bottom
            if NC > 100
              hmin = min(Ce(comp,3));
              H = max(Ce(comp,3))-hmin;
              if H > 5 && hmin < bottom+5
                I = Ce(NearSets,3) < hmin+0.5;
                NearSets = NearSets(I);
                I = Trunk(NearSets);
                J = Forb(NearSets);
                trunk = NearSets(I); % "Trunk" sets
                forb = NearSets(J); % "Forb" sets
              end
            end

            % Determine the closest sets for "trunk"
            if ~isempty(trunk)
              d = pdist2(Ce(comp,:),Ce(trunk,:));
              if NC == 1 && length(trunk) == 1
                dt = d; % the minimum distance
                IC = 1; % the set in component to be connected
                IT = 1; % the set in "trunk" to be connected
              elseif NC == 1
                [dt,IT] = min(d);
                IC = 1;
              elseif length(trunk) == 1
                [dt,IC] = min(d);
                IT = 1;
              else
                [d,IC] = min(d);
                [dt,IT] = min(d);
                IC = IC(IT);
              end
            else
              dt = 700;
            end

            % Determine the closest sets for "forb"
            if ~isempty(forb)
              d = pdist2(Ce(comp,:),Ce(forb,:));
              df = min(d);
              if length(df) > 1
                df = min(df);
              end
            else
              df = 1000;
            end

            % Determine what to do with the component
            if (dt > 12 && dt < 100) || (NC < Cmin && dt > 0.5 && dt < 10)
              % Remove small isolated component
              Forb(comp) = true;
              Other(comp) = false;
              NonClassified(i) = false;
            elseif 3*df < dt || (df < dt && df > 0.25)
              % Join the component to "Forb"
              Forb(comp) = true;
              Other(comp) = false;
              NonClassified(i) = false;
            elseif (df == 1000 && dt == 700) || dt > k*inputs.PatchDiam1
              % Isolated component, do nothing
            else
              % Join to "Trunk"
              I = comp(IC);
              J = trunk(IT);
              Other(comp) = false;
              Trunk(comp) = true;
              NonClassified(i) = false;
              % make the connection
              Nei{I} = [Nei{I}; J];
              Nei{J} = [Nei{J}; I];
            end
          end
        end
      end
    end
    first_round = false;
    % If "Other" has decreased, do another check with same "distance"
    if nnz(NonClassified) < npre
      again = true;
      npre = nnz(NonClassified);
    else
      again = false;
    end
  end
  k = k+k0; % increase the cell size of the nearby search space
  Cmin = 3*Cmin; % increase the acceptable component size
end
Forb(Base) = false;
cover.neighbor = Nei;

end % End of function

function segment = segments(cover,Base,Forb)

% ---------------------------------------------------------------------
% SEGMENTS.M        Segments the covered point cloud into branches.
%
% Version 2.10
% Latest update     16 Aug 2017
%
% Copyright (C) 2013-2017 Pasi Raumonen
% ---------------------------------------------------------------------

% Segments the tree into branches and records their parent-child-relations. 
% Bifurcations are recognized by studying connectivity of a "study"
% region moving along the tree. In case of multiple connected components 
% in "study", the components are classified as the continuation and branches.
%
% Inputs:
% cover         Cover sets
% Base          Base of the tree
% Forb          Cover sets not part of the tree
%
% Outputs:
% segment       Structure array containing the followin fields:
%   segments          Segments found, (n_seg x 1)-cell, each cell contains a cell array the cover sets
%   ParentSegment     Parent segment of each segment, (n_seg x 1)-vector,
%                       equals to zero if no parent segment
%   ChildSegment      Children segments of each segment, (n_seg x 1)-cell

Nei = cover.neighbor;
nb = size(Nei,1);           % The number of cover sets
a = max([200000 nb/100]);   % Estimate for maximum number of segments
SBas = cell(a,1);           % The segment bases found
Segs = cell(a,1);           % The segments found
SPar = zeros(a,2,'uint32'); % The parent segment of each segment
SChi = cell(a,1);           % The children segments of each segment

% Initialize SChi
SChi{1} = zeros(5000,1,'uint32');
C = zeros(200,1);
for i = 2:a
    SChi{i} = C;
end
NChi = zeros(a,1);      % Number of child segments found for each segment

Fal = false(nb,1);      % Logical false-vector for cover sets
s = 1;                  % The index of the segment under expansion
b = s;                  % The index of the latest found base

SBas{s} = Base;
Seg = cell(1000,1);    % The cover set layers in the current segment
Seg{1} = Base;

ForbAll = Fal;       % The forbidden sets
ForbAll(Forb) = true;
ForbAll(Base) = true;
Forb = ForbAll;      % The forbidden sets for the segment under expansion

Continue = true; % True as long as the component can be segmented further 
NewSeg = true;   % True if the first Cut for the current segment
nl = 1;          % The number of cover set layers currently in the segment

% Segmenting stops when there are no more segments to be found
while Continue && (b < nb)
    
    % Update the forbidden sets
    Forb(Seg{nl}) = true;
    
    % Define the study
    Cut = define_cut(Nei,Seg{nl},Forb,Fal);
    CutSize = length(Cut);
    
    if NewSeg
        NewSeg = false;
        ns = min(CutSize,6);
    end
    
    % Define the components of cut and study regions
    if CutSize > 0
        CutComps = cut_components(Nei,Cut,CutSize,Fal,Fal);
        nc = size(CutComps,1);
        if nc > 1
            [StudyComps,Bases,CompSize,Cont,BaseSize] = ...
                study_components(Nei,ns,Cut,CutComps,Forb,Fal,Fal);
            nc = length(Cont);
        end
    else
        nc = 0;
    end
    
    % Classify study region components
    if nc == 1
        % One component, continue expansion of the current segment
        nl = nl+1;
        if size(Cut,2) > 1
            Seg{nl} = Cut';
        else
            Seg{nl} = Cut;
        end
    elseif nc > 1
        % Classify the components of the Study region
        Class = component_classification(CompSize,Cont,BaseSize,CutSize);
        
        for i = 1:nc
            if Class(i) == 1
                Base = Bases{i};
                ForbAll(Base) = true;
                Forb(StudyComps{i}) = true;
                J = Forb(Cut);
                Cut = Cut(~J);
                b = b+1;
                SBas{b} = Base;
                SPar(b,:) = [s nl];
                NChi(s) = NChi(s)+1;
                SChi{s}(NChi(s)) = b;
            end
        end
        
        % Define the new cut.
        % If the cut is empty, determine the new base
        if isempty(Cut)
            Segs{s} = Seg(1:nl);
            S = vertcat(Seg{1:nl});
            ForbAll(S) = true;

            if s < b
                s = s+1;
                Seg{1} = SBas{s};
                Forb = ForbAll;
                NewSeg = true;
                nl = 1;
            else
                Continue = false;
            end
        else
            if size(Cut,2) > 1
                Cut = Cut';
            end
            nl = nl+1;
            Seg{nl} = Cut;
        end
    
    else
        % If the study region has zero size, then the current segment is
        % complete and determine the base of the next segment
        Segs{s} = Seg(1:nl);
        S = vertcat(Seg{1:nl});
        ForbAll(S) = true;
        
        if s < b
            s = s+1;
            Seg{1} = SBas{s};
            Forb = ForbAll;
            NewSeg = true;
            nl = 1;
        else
            Continue = false;
        end
    end
end
Segs = Segs(1:b);
SPar = SPar(1:b,:);
schi = SChi(1:b);

% Define output
SChi = cell(b,1);
for i = 1:b
    if NChi(i) > 0
        SChi{i} = uint32(schi{i}(1:NChi(i)));
    else
        SChi{i} = zeros(0,1,'uint32');
    end
    S = Segs{i};
    for j = 1:size(S,1)
        S{j} = uint32(S{j});
    end
    Segs{i} = S;
end
clear Segment
segment.segments = Segs;
segment.ParentSegment = SPar;
segment.ChildSegment = SChi;

end % End of the main function


% Define subfunctions

function Cut = define_cut(Nei,CutPre,Forb,Fal)

% Defines the "Cut" region
Cut = vertcat(Nei{CutPre});
Cut = unique_elements(Cut,Fal);
I = Forb(Cut);
Cut = Cut(~I);
end % End of function 


function [Components,CompSize] = cut_components(Nei,Cut,CutSize,Fal,False)

% Define the connected components of the Cut
if CutSize == 1
    % Cut is connected and therefore Study is also
    CompSize = 1;
    Components = cell(1,1);
    Components{1} = Cut;
elseif CutSize == 2
    I = Nei{Cut(1)} == Cut(2);
    if any(I)
        Components = cell(1,1);
        Components{1} = Cut;
        CompSize = 1;
    else
        Components = cell(2,1);
        Components{1} = Cut(1);
        Components{2} = Cut(2);
        CompSize = [1 1];
    end
elseif CutSize == 3
    I = Nei{Cut(1)} == Cut(2);
    J = Nei{Cut(1)} == Cut(3);
    K = Nei{Cut(2)} == Cut(3);
    if any(I)+any(J)+any(K) >= 2
        CompSize = 1;
        Components = cell(1,1);
        Components{1} = Cut;
    elseif any(I)
        Components = cell(2,1);
        Components{1} = Cut(1:2);
        Components{2} = Cut(3);
        CompSize = [2 1];
    elseif any(J)
        Components = cell(2,1);
        Components{1} = Cut([1 3]');
        Components{2} = Cut(2);
        CompSize = [2 1];
    elseif any(K)
        Components = cell(2,1);
        Components{1} = Cut(2:3);
        Components{2} = Cut(1);
        CompSize = [2 1];
    else
        CompSize = [1 1 1];
        Components = cell(3,1);
        Components{1} = Cut(1);
        Components{2} = Cut(2);
        Components{3} = Cut(3);
    end
else
    Components = cell(CutSize,1);
    CompSize = zeros(CutSize,1);
    Comp = zeros(CutSize,1);
    Fal(Cut) = true;
    nc = 0;      % number of components found
    m = Cut(1);
    i = 0;
    while i < CutSize
        Added = Nei{m};
        I = Fal(Added);
        Added = Added(I);
        a = length(Added);
        Comp(1) = m;
        Fal(m) = false;
        t = 1;
        while a > 0
            Comp(t+1:t+a) = Added;
            Fal(Added) = false;
            t = t+a;
            Ext = vertcat(Nei{Added});
            Ext = unique_elements(Ext,False);
            I = Fal(Ext);
            Added = Ext(I);
            a = length(Added);
        end
        i = i+t;
        nc = nc+1;
        Components{nc} = Comp(1:t);
        CompSize(nc) = t;
        if i < CutSize
            J = Fal(Cut);
            m = Cut(J);
            m = m(1);
        end
    end
    Components = Components(1:nc);
    CompSize = CompSize(1:nc);
end

end % End of function


function [Components,Bases,CompSize,Cont,BaseSize] = ...
    study_components(Nei,ns,Cut,CutComps,Forb,Fal,False)

% Define Study as a cell-array
Study = cell(ns,1);
StudySize = zeros(ns,1);
Study{1} = Cut;
StudySize(1) = length(Cut);
if ns >= 2
    N = Cut;
    i = 1;
    while i < ns
        Forb(N) = true;
        N = vertcat(Nei{N});
        N = unique_elements(N,Fal);
        I = Forb(N);
        N = N(~I);
        if ~isempty(N)
            i = i+1;
            Study{i} = N;
            StudySize(i) = length(N);
        else
            Study = Study(1:i);
            StudySize = StudySize(1:i);
            i = ns+1;
        end
    end
end

% Define study as a vector
ns = length(StudySize);
studysize = sum(StudySize);
study = vertcat(Study{:});

% Determine the components of study
nc = size(CutComps,1);
i = 1; % index of cut component
j = 0; % number of elements attributed to components
k = 0; % number of study components
Fal(study) = true;
Components = cell(nc,1);
CompSize = zeros(nc,1);
Comp = zeros(studysize,1);
while i <= nc
    C = CutComps{i};
    while j < studysize
        a = length(C);
        Comp(1:a) = C;
        Fal(C) = false;
        if a > 1
            Add = unique_elements(vertcat(Nei{C}),False);
        else
            Add = Nei{C};
        end
        t = a;
        I = Fal(Add);
        Add = Add(I);
        a = length(Add);
        while a > 0
            Comp(t+1:t+a) = Add;
            Fal(Add) = false;
            t = t+a;
            Add = vertcat(Nei{Add});
            Add = unique_elements(Add,False);
            I = Fal(Add);
            Add = Add(I);
            a = length(Add);
        end
        j = j+t;
        k = k+1;
        Components{k} = Comp(1:t);
        CompSize(k) = t;
        if j < studysize
            C = zeros(0,1);
            while i < nc && isempty(C)
                i = i+1;
                C = CutComps{i};
                J = Fal(C);
                C = C(J);
            end
            if i == nc && isempty(C)
                j = studysize;
                i = nc+1;
            end
        else
            i = nc+1;
        end
    end
    Components = Components(1:k);
    CompSize = CompSize(1:k);
end

% Determine BaseSize and Cont
Cont = true(k,1);
BaseSize = zeros(k,1);
Bases = cell(k,1);
if k > 1
    Forb(study) = true;
    Fal(study) = false;
    Fal(Study{1}) = true;
    for i = 1:k
        % Determine the size of the base of the components
        Set = unique_elements([Components{i}; Study{1}],False);
        False(Components{i}) = true;
        I = False(Set)&Fal(Set);
        False(Components{i}) = false;
        Set = Set(I);
        Bases{i} = Set;
        BaseSize(i) = length(Set);
    end
    Fal(Study{1}) = false;
    Fal(Study{ns}) = true;
    Forb(study) = true;
    for i = 1:k
        % Determine if the component can be extended
        Set = unique_elements([Components{i}; Study{ns}],False);
        False(Components{i}) = true;
        I = False(Set)&Fal(Set);
        False(Components{i}) = false;
        Set = Set(I);
        if ~isempty(Set)
            N = vertcat(Nei{Set});
            N = unique_elements(N,False);
            I = Forb(N);
            N = N(~I);
            if isempty(N)
                Cont(i) = false;
            end
        else
            Cont(i) = false;
        end
    end
end

end % End of function


function Class = component_classification(CompSize,Cont,BaseSize,CutSize)

% Classifies study region components:
% Class(i) == 0 continuation
% Class(i) == 1 branch

nc = size(CompSize,1);
StudySize = sum(CompSize);
Class = ones(nc,1);     % true if a component is a branch to be further segmented
ContiComp = 0;
% Simple initial classification
for i = 1:nc
    if BaseSize(i) == CompSize(i) && ~Cont(i)
        % component has no expansion, not a branch
        Class(i) = 0;
    elseif BaseSize(i) == 1 && CompSize(i) <= 2 && ~Cont(i)
        % component has very small expansion, not a branch
        Class(i) = 0;
    elseif BaseSize(i)/CutSize < 0.05 && 2*BaseSize(i) >= CompSize(i) && ~Cont(i)
        % component has very small expansion or is very small, not a branch
        Class(i) = 0;
    elseif CompSize(i) <= 3 && ~Cont(i)
        % very small component, not a branch
        Class(i) = 0;
    elseif BaseSize(i)/CutSize >= 0.7 || CompSize(i) >= 0.7*StudySize
        % continuation of the segment
        Class(i) = 0;
        ContiComp = i;
    else
        % Component is probably a branch
    end
end

Branches = Class == 1;
if ContiComp == 0 && any(Branches)
    Ind = (1:1:nc)';
    Branches = Ind(Branches);
    [~,I] = max(CompSize(Branches));
    Class(Branches(I)) = 0;
end

end % End of function

function pmdistance = point_model_distance(P,cylinder)

% ---------------------------------------------------------------------
% POINT_MODEL_DISTANCE.M    Computes the distances of the points to the 
%                               cylinder model
%
% Version 2.1.1
% Latest update     8 Oct 2021
%
% Copyright (C) 2015-2021 Pasi Raumonen
% ---------------------------------------------------------------------

% Changes from version 2.1.0 to 2.1.1, 8 Oct 2021:  
% 1) Changed the determinationa NE, the number of empty edge layers, so 
%     that is now limited in size, before it is given as input for 
%     cubical_partition function.

% Changes from version 2.0.0 to 2.1.0, 26 Nov 2019:  
% 1) Bug fix: Corrected the computation of the output at the end of the
%    code so that trees without branches are computed correctly.

% Cylinder data
Rad = cylinder.radius;
Len = cylinder.length;
Sta = cylinder.start;
Axe = cylinder.axis;
BOrd = cylinder.BranchOrder;

% Select randomly 25 % or max one million points for the distance comput.
np0 = size(P,1);
a = min(0.25*np0,1000000);
I = logical(round(0.5/(1-a/np0)*rand(np0,1)));
P = P(I,:);

% Partition the points into cubes 
L = 2*median(Len);
NE = max(3,min(10,ceil(max(Len)/L)))+3;
[Partition,~,Info] = cubical_partition(P,L,NE);
Min = Info(1:3);
EL = Info(7);
NE = Info(8);

% Calculates the cube-coordinates of the starting points
CC = floor([Sta(:,1)-Min(1) Sta(:,2)-Min(2) Sta(:,3)-Min(3)]/EL)+NE+1;

% Compute the number of cubes needed for each starting point
N = ceil(Len/L);

% Correct N so that cube indexes are not too small or large
I = CC(:,1) < N+1;
N(I) = CC(I,1)-1;
I = CC(:,2) < N+1;
N(I) = CC(I,2)-1;
I = CC(:,3) < N+1;
N(I) = CC(I,3)-1;
I = CC(:,1)+N+1 > Info(4);
N(I) = Info(4)-CC(I,1)-1;
I = CC(:,2)+N+1 > Info(5);
N(I) = Info(5)-CC(I,2)-1;
I = CC(:,3)+N+1 > Info(6);
N(I) = Info(6)-CC(I,3)-1;

% Calculate the distances to the cylinders
n = size(Rad,1);
np = size(P,1);
Dist = zeros(np,2); % Distance and the closest cylinder of each points
Dist(:,1) = 2; % Large distance initially
Points = zeros(ceil(np/10),1,'int32'); % Auxiliary variable
Data = cell(n,1);
for i = 1:n
  Par = Partition(CC(i,1)-N(i):CC(i,1)+N(i),CC(i,2)-N(i):CC(i,2)+N(i),...
    CC(i,3)-N(i):CC(i,3)+N(i));
  if N(i) > 1
    S = cellfun('length',Par);
    I = S > 0;
    S = S(I);
    Par = Par(I);
    stop = cumsum(S);
    start = [0; stop]+1;
    for k = 1:length(stop)
      Points(start(k):stop(k)) = Par{k}(:);
    end
    points = Points(1:stop(k));
  else
    points = vertcat(Par{:});
  end
  [d,~,h] = distances_to_line(P(points,:),Axe(i,:),Sta(i,:));
  d = abs(d-Rad(i));
  Data{i} = [d h double(points)];
  I = d < Dist(points,1);
  J = h >= 0;
  K = h <= Len(i);
  L = d < 0.5;
  M = I&J&K&L;
  points = points(M);
  Dist(points,1) = d(M);
  Dist(points,2) = i;
end

% Calculate the distances to the cylinders for points not yet calculated
% because they are not "on side of cylinder
for i = 1:n
  if ~isempty(Data{i})
    d = Data{i}(:,1);
    h = Data{i}(:,2);
    points = Data{i}(:,3);
    I = d < Dist(points,1);
    J = h >= -0.1 & h <= 0;
    K = h <= Len(i)+0.1 & h >= Len(i);
    L = d < 0.5;
    M = I&(J|K)&L;
    points = points(M);
    Dist(points,1) = d(M);
    Dist(points,2) = i;
  end
end

% Select only the shortest 95% of distances for each cylinder
N = zeros(n,1);
O = zeros(np,1);
for i = 1:np
  if Dist(i,2) > 0
    N(Dist(i,2)) = N(Dist(i,2))+1;
    O(i) = N(Dist(i,2));
  end
end
Cyl = cell(n,1);
for i = 1:n
  Cyl{i} = zeros(N(i),1);
end
for i = 1:np
  if Dist(i,2) > 0
    Cyl{Dist(i,2)}(O(i)) = i;
  end
end
DistCyl = zeros(n,1); % Average point distance to each cylinder
for i = 1:n
  I = Cyl{i};
  m = length(I);
  if m > 19 % select the smallest 95% of distances
    d = sort(Dist(I,1));
    DistCyl(i) = mean(d(1:floor(0.95*m)));
  elseif m > 0
    DistCyl(i) = mean(Dist(I,1));
  end
end

% Define the output
pmdistance.CylDist = single(DistCyl);
pmdistance.median = median(DistCyl(:,1));
pmdistance.mean = mean(DistCyl(:,1));
pmdistance.max = max(DistCyl(:,1));
pmdistance.std = std(DistCyl(:,1));

T = BOrd == 0;
B1 = BOrd == 1;
B2 = BOrd == 2;
B = DistCyl(~T,1);
T = DistCyl(T,1);
B1 = DistCyl(B1,1);
B2 = DistCyl(B2,1);

pmdistance.TrunkMedian = median(T);
pmdistance.TrunkMean = mean(T);
pmdistance.TrunkMax = max(T);
pmdistance.TrunkStd = std(T);

if ~isempty(B)
  pmdistance.BranchMedian = median(B);
  pmdistance.BranchMean = mean(B);
  pmdistance.BranchMax = max(B);
  pmdistance.BranchStd = std(B);
else
  pmdistance.BranchMedian = 0;
  pmdistance.BranchMean = 0;
  pmdistance.BranchMax = 0;
  pmdistance.BranchStd = 0;
end

if ~isempty(B1)
  pmdistance.Branch1Median = median(B1);
  pmdistance.Branch1Mean = mean(B1);
  pmdistance.Branch1Max = max(B1);
  pmdistance.Branch1Std = std(B1);
else
  pmdistance.Branch1Median = 0;
  pmdistance.Branch1Mean = 0;
  pmdistance.Branch1Max = 0;
  pmdistance.Branch1Std = 0;
end

if ~isempty(B2)
  pmdistance.Branch2Median = median(B2);
  pmdistance.Branch2Mean = mean(B2);
  pmdistance.Branch2Max = max(B2);
  pmdistance.Branch2Std = std(B2);
else
  pmdistance.Branch2Median = 0;
  pmdistance.Branch2Mean = 0;
  pmdistance.Branch2Max = 0;
  pmdistance.Branch2Std = 0;
end
end