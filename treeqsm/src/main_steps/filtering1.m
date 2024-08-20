% This file is part of TREEQSM.
% 
% TREEQSM is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% TREEQSM is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with TREEQSM.  If not, see <http://www.gnu.org/licenses/>.



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
