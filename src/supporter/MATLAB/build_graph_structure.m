function graph = build_graph_structure(pts, k, edge_weight_mode)
    % Compute the k-nearest neighbors graph structure.
    % This function is created by Loic Landrieu on
    % https://github.com/loicland/point-cloud-regularization/
    %
    % INPUT
    % pts: a matrix of n x 3 point cloud 
    % k: number of nodes (default is 10)
    % edge_weight_mode: weighting mode of the edges
    %   c = 0 : constant weight (default)
    %   c > 0 : linear weight w = 1/(d/d0 + c) 
    %   c < 0 : exponential weight w = exp(-d/(c * d0))
    %
    % OUTPUT
    % graph: a structure with the following fields:
    %   .XYZ : coordinates of each point
    %   .source: index of the source vertices constituting the edges
    %   .target: index of the target vertices constituting the edges
    %   .edge_weight: the weights of the edges
    
    if nargin < 2
        k = 10;     % default number of neighbors
    end
    if nargin < 3
        edge_weight_mode = 0;   % default edge weight mode
    end
    
    % Initial graph structure
    graph= struct;
    graph.XYZ = pts;
    n_point = size(graph.XYZ,1);
    
    % Compute full adjacency graph
    [neighbors, distance] = knnsearch(graph.XYZ, graph.XYZ, 'K', k + 1);
    neighbors = neighbors(:, 2:end);    % exclude self from neighbors
    distance = distance(:, 2:end);
    d0 = mean(distance(:)); % average distance for normalization
    
    % Create source and target edges
    source = reshape(repmat(1:n_point, [k 1]), [1, k * n_point])';
    target = reshape(neighbors', [1, k * n_point])';
    
    % Compute edge weights based on the specified mode
    edge_weight = ones(size(distance));
    if edge_weight_mode > 0
        edge_weight = 1 ./ (distance / d0 + edge_weight_mode);
    elseif edge_weight_mode < 0
        edge_weight = exp(distance / (d0 * edge_weight_mode));
    end
    edge_weight = reshape(edge_weight', [1, k * n_point])';
    
    % Prune edges based on distance statistics
    dt = mean(distance, 2) + std(distance, 0, 2);
    prune = bsxfun(@gt, distance', dt')';  % identify edges to prune
    pruned = reshape(prune', [1, k * n_point])';
    selfedge = source == target;  % identify self-edges
    to_remove = selfedge + pruned;  % combine self-edges and pruned edges
    
    % Remove self edges and pruned edges
    source = source(~to_remove) - 1;  % adjust for MATLAB 1-based indexing
    target = target(~to_remove) - 1;
    edge_weight = edge_weight(~to_remove);
    
    % Symetrizing the graph to avoid duplicate edges
    double_edges_coord  = [[source; target], [target; source]];
    double_edges_weight = [edge_weight; edge_weight];                 
    [edges_coord, order] = unique(double_edges_coord, 'rows');
    edges_weight = double_edges_weight(order);
    
    % Fill the graph structure
    graph.source = int32(edges_coord(:, 1));
    graph.target = int32(edges_coord(:, 2));
    graph.edge_weight = single(edges_weight);
end
