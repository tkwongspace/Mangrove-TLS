function [assignment, T] = alpha_expansion(initial_p, graph, fidelity, lambda)
    % Alpha expansion algorithm to solve penalization by the Potts penalty.
    % This function is created by Loic Landrieu on
    % https://github.com/loicland/point-cloud-regularization/
    %
    % INPUT
    % inital_p: initial labeling to regularize
    % graph: the adjacency structure
    % fidelity: which fidelity fucntion to use (default = 0)
    %	0 : linear
    %	1 : quadratic  
    %	2 : KL with 0.05 uniform smoothing
    %	3 : loglinear with 0.05 uniform smoothing
    % lambda: regularization strength (default = 1)
    %
    % OUTPUT
    % assignment: the regularized labeling
    % T: computing time
    %
    % /// COPYRIGHT INFORMATION ///
    % GCMex function embedded in this function is written by Shai Bagon.
    % Citations to use this function:
    % [1] Yuri Boykov, Olga Veksler, Ramin Zabih. Fast approximate energy 
    %       minimization via graph cuts. IEEE transactions on Pattern 
    %       Analysis and Machine Intelligence. 2001, 20(12): 1222-1239.
    %
    % [2] Vladimir Kolmogorov & Ramin Zabih. What energy functions can be 
    %       minimized via graph cuts? IEEE Transactions on Pattern 
    %       Analysis and Machine Intelligence. 2004, 26(2): 147-159.
    %
    % [3] Yuri Boykov & Vladimir Kolmogorov. An Experimental Comparison of 
    %       Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision. 
    %       IEEE Transactions on Pattern Analysis and Machine Intelligence.
    %       2004, 26(9): 1124-1137.
    %
    % [4] Shai Bagon. Matlab Wrapper for Graph Cut. 2006. 
    %       https://github.com/shaibagon/GCMex.
    
    % Default parameter values
    if nargin < 3
        fidelity = 0;  % default fidelity function
    end
    if nargin < 4
        lambda = 1;  % default regularization strength
    end
    % if nargin < 5
    %     maxIte = 30;  % default maximum iterations
    % end
    
    smoothing = 0.05; % smoothing paremeter
    nbNode = size(initial_p,1);
    nClasses = size(initial_p,2);
    
    % Create pairwise potential matrix
    pairwise = sparse(double(graph.source + 1), ...
        double(graph.target + 1), ...
        double(graph.edge_weight * lambda));
    
    % Ensure pairwise matrix matches the number of nodes
    % pairwise = padarray(pairwise, ...
    %     [nbNode - size(pairwise, 1), nbNode - size(pairwise, 2)], ...
    %     0, 'post');
    mpairwise = size(pairwise, 1);
    npairwise = size(pairwise, 2);
    if mpairwise ~= nbNode | npairwise ~= nbNode
        dm = nbNode - mpairwise;
        dn = nbNode - npairwise;
        pairwise = [pairwise, zeros(mpairwise, dn)];
        pairwise = [pairwise, zeros(dm, size(pairwise, 2))];
    end

    % Compute unary potentials based on the fidelity function
    switch fidelity
        case 0  % linear
            unary = squeeze(sum( ...
                ((repmat(1:nClasses, [size(initial_p, 1), 1, nClasses]) == ...
                permute(repmat(1:nClasses, [size(initial_p, 1), 1, nClasses]), [1, 3, 2])) - ...
                repmat(initial_p, [1, 1, size(initial_p, 2)])).^2, 2));
        case 1  % quadratic
            unary = -initial_p;
        case 2  % KL divergence
            smoothObs = repmat(smoothing / nClasses + (1 - smoothing) * initial_p, ...
                [1, 1, nClasses]);
            smoothAssi = smoothing / nClasses + (1 - smoothing) * ...
                        (repmat(1:nClasses, [nbNode, 1, nClasses]) == ...
                        permute(repmat(1:nClasses, [nbNode, 1, nClasses]), [1, 3, 2]));
            unary = -squeeze(sum(smoothObs .* (log(smoothAssi)), 2));
        case 3  % loglinear
            smoothObs = smoothing / nClasses + (1 - smoothing) * initial_p;
            unary = -log(smoothObs); 
    end
    
    % Normalize unary potentials
    unary = bsxfun(@minus, unary, min(unary, [], 2));
    unary = unary ./ mean(unary(:)); % normalize by mean
    
    % Label cost matrix
    labelcost  = single(1 * (ones(nClasses) - eye(nClasses)));
    
    % Initial labeling
    [~, labelInit] = max(initial_p, [], 2);
    labelInit = labelInit - 1;  % convert to zero-based indexing
    clear I J V
    
    % Run the alpha expansion algorithm
    tic;
    [assignment] = GCMex(double(labelInit), ...
        (unary'), (pairwise), single(labelcost), true);
    assignment = assignment + 1;  % convert back to one-based indexing
    T = toc;  % measure elapsed time
end