function [wood, leaf, label_idx] = MangroveSeg(points, ft_threshold, ...
    qsm_input, regu, plot)
    % INPUT
    % points: point cloud (pointcloud)
    % ft_threshold: feature threshold, suggest using 0.125 or so (number)
    % qsm_input: input structure for TreeQSM (struct)
    % regu: if apply regularization using alpha expansion (1 or other)
    % plot: if plot results in the end (1 or other)
    % 
    % OUTPUT
    % wood: points of woods
    % leaf: points of leaves
    % pts_idx: point index with 1 for wood and 0 for leaf
    
    % Function created by Di Wang, di.wang@aalto.fi
    % URL: https://github.com/dwang520/LeWoS
    
    disp('----------------------')
    disp(">> Mangrove Woods/Leaves Segmentation Start...")

    %% 1. Calculate normals    
    % using 10 points for local plane fitting
    normals = pcnormals(points,10);
    
    %% 2. Segment initial regions using graph-based region growing
    disp(">> Graph-based region growing for segmentation..")

    k = 10;  % number of neighbors for region growing
    Label = GraphRG(points.Location, normals(:,3), k, ft_threshold);
    
    % Group points into segments (Seg, cell)
    t = accumarray(Label, (1:length(Label))', [], @(x) {x});
    Seg = cellfun(@(x) points.Location(x, :), t, 'UniformOutput', false);
    
    % Filter segments based on size (threshold = 10 points)
    lb = cellfun(@(x) size(x, 1), Seg);
    Seg_reserve = Seg(lb <= 10);      % no need for further segmentation
    Seg_processing = Seg(lb > 10);    % wait for further segmentation
    clear Seg t lb Label normals
    
    %% 3. Recursive segmentation
    % segment each segments iteratively until it doesnt change anymore.
    h = waitbar(0, ">> Recursive segmentation..");
    
    for ii = 1:10   % limit to 10 iterations to avoid infinite loops
        TMP = cell(length(Seg_processing),1);
        indx = false(length(Seg_processing),1);
    
        % parallel work to identify if segmentation needs further works
        parfor i = 1:length(Seg_processing)
            % groupping points in each segment
            points_n = Seg_processing{i};
            normals = pcnormals(pointCloud(points_n(:, 1:3)), 10);
            Label = GraphRG(points_n, normals(:, 3), k, ft_threshold);
    
            % mark each segment whether process is finished
            if max(Label)==1
                % if only 1 group is left
                indx(i) = true;
            else
                % if multiple groups are identified, 
                % get ready for further segmentation
                t = accumarray(Label, (1:length(Label))', [], @(x) {x});
                TMP{i} = cellfun(@(x) points_n(x,:), t, 'UniformOutput', false);
            end
        end
        
        Seg = cat(1, TMP{:});
        unchange = Seg_processing(indx);
    
        waitbar(ii / 10, h, sprintf("Segmentation %d of %d...", ii, 10));

        if sum(indx) == length(Seg_processing) || ii == 10
            Seg_final = [Seg; Seg_reserve; unchange];
            break;
        else
            lb = cellfun(@(x) size(x,1),Seg);        
            Seg_processing = Seg(lb>10);        
            Seg_reserve = [Seg_reserve; unchange; Seg(lb <= 10)];
        end
    end
    
    clear Seg_reserve unchange Seg_processing Seg lb TMP indx h
    
    % The final segmentation results after recursive segmentation is stored in Seg_final (cell)
    
    % %% 3.1 visualize segmentation results
    % cmap = hsv(length(Seg_final));
    % col = arrayfun(@(i) repmat(cmap(i, :), size(Seg_final{i}, 1), 1), ...
    %     (1:length(Seg_final))', 'UniformOutput', false);
    % 
    % figure;pcshow(cell2mat(Seg_final),cell2mat(col));grid off;
    % 
    % clear cmap pz col ww
    
    %% 4. Further split each segment into individual branches using TreeQSM
    % The z value of normal vectors itself 
    % is not enough to segment points into branches. 
    % In some case, it forms a large co-planar segment. 
    % We use the method from TreeQSM to further split segment. 
    % https://github.com/InverseTampere/TreeQSM
    % also
    % https://github.com/InverseTampere/Vessel-Segmentation

    disp(">> Spliting branches with TreeQSM..")
    
    % Split segments
    Seg_post = cell(length(Seg_final),1);
    parfor i = 1:length(Seg_final)    
        pts = Seg_final{i};
    
        if size(pts, 1) > 100 && (max(pts(:, 3)) - min(pts(:, 3))) > 1
            % generate cover sets
            cover1 = cover_sets(pts,qsm_input);
    
            if length(cover1.ball) > 2
                % find a base cover
                Base = find(pts(cover1.center, 3) == min(pts(cover1.center, 3)));
                Forb = false(length(cover1.center), 1);
                % do the segmentation
                segment1 = segments(cover1, Base, Forb);
    
                Seg_tmp = cell(length(segment1.segments), 1);
                for j = 1:length(segment1.segments)
                    ids = cell2mat(cover1.ball(cell2mat(segment1.segments{j})));
                    Seg_tmp{j} = pts(ids, :);
                end
                Seg_post{i} = Seg_tmp;
            else
                Seg_post{i} = {pts};
            end
        else
            Seg_post{i} = {pts};
        end
    end
    Seg_final = cat(1, Seg_post{:});
    clear Seg_post
    
    % Seg_final is the final segmentaion results after all processing
    
    %shut down parallel pool to save memory    
    poolobj = gcp('nocreate');
    delete(poolobj);
    
    % ///////// Segmentation is done above////////////////////////////
    % ///////// Below start find branch segments/////////////////////
    
    %% 5. Probability estimation for each point
    % two thresholds, Linearity threshold and size threshold are needed 
    % to identify those segmentation belonging to branches. 
    % Instead of specifying hard thresholds, 
    % we test a range of values and count for the frequency of a point 
    % being identified as wood. 
    % The output of this step is a probability distribution.
    
    disp(">> Estimating probability for each point..")

    % Calculate linearity of each segment
    linearity = Cal_Linearity(Seg_final);
    % calculate size of each segment
    sl = cellfun(@(x) size(x, 1), Seg_final);
    
    % Prepare point-level data
    % upwrap segments to points, so we can operate on point level
    LL = cell(length(Seg_final), 1);
    SL = cell(length(Seg_final), 1);
    for i = 1:length(Seg_final)
        P = Seg_final{i};
        LL{i} = repmat(linearity(i), size(P, 1), 1);
        SL{i} = repmat(sl(i), size(P, 1), 1);
    end
    LiPts = cell2mat(LL);
    SzPts = cell2mat(SL);
    Pts = cell2mat(Seg_final);
    
    % Threshold testing
    Lthres_list = 0.70:0.02:0.95;
    Sthres_list = 10:2:50;
    allc = combvec(Lthres_list,Sthres_list)';
    
    Freq = zeros(size(Pts, 1), 1);
    for i = 1:size(allc, 1)
        % find wood points based on two thresholds
        ia = LiPts >= allc(i, 1) & SzPts >= allc(i, 2);
        % count the frequency of being identified as wood
        Freq = Freq + ia;
    end
    
    % Probability that a point is wood
    Pli = Freq / size(allc, 1);
    
    % ///////// Probability estimation is done above////////////////////////////
    % ///////// Below start regularization (label smoothing)/////////////////////
    
    %% 6. Regularization using Alpha-Expansion
    % ALPHA-EXPANSION method from
    % https://github.com/loicland/point-cloud-regularization
    % (it has a native C++ implementaion I think)
    % to regularize point labels to make the final prediction spatially smooth.
    % The method is also based on graph energy optimization, 
    % and requires a probability as input. 
    % Our above method is naturally suitble for this.
    
    % build adjacent graph (similar to the one in "GraphRG")
    graph = build_graph_structure(Pts, 20, 0);
    initial_classif = single([Pli, 1 - Pli]);  % initial class probability

    if regu == 1
        % alpha expansion method, output is the regularized label per point
        [labeled_pts, ~] = alpha_expansion(initial_classif, graph, 0, 1);
        disp(">> Regularization finished.")
    else
        % record the label without regularization (directly from "Pli")
        [~, labeled_pts] = max(initial_classif, [], 2);
    end
    
    % ///////// Regularization is done above////////////////////////////
    % ///////// Below prepare final results/////////////////////
    
    %% 7. Restore original order and prepare final labels
    % wood label ->1 , leaf label -> 0
    
    idx = knnsearch(Pts, points.Location);    
    label_idx = labeled_pts(idx);
    label_idx(label_idx ~= 1) = 0;

    wood = Pts(labeled_pts == 1, :);
    leaf = Pts(labeled_pts ~= 1, :);

    disp(">> Tree segmentation finished.")

    %% 8. Visualization
    if plot == 1        
        figure('Name', 'Woods & Leaves Segmentation', ...
            'units', 'normalized', ...
            'outerposition', [0 0 1 1]);

        pcshow(wood, repmat([0.4471, 0.3216, 0.1647], size(wood, 1), 1));
        hold on;
        pcshow(leaf, repmat([0.2667, 0.5686, 0.1961], size(leaf, 1), 1));
        hold off;
        grid off;
        xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
        title('Scanned Tree');
    end
end
