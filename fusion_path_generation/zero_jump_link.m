function paths = zero_jump_link(frames)
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2015, Georgia Gkioxari
%
% This file is part of the Action Tubes code and is available
% under the terms of the Simplified BSD License provided in
% LICENSE. Please retain this notice and LICENSE if you use
% this file (or any portion of it) in your project.

% at each iteration in the while loop, a new path is generated and the boxes, scores and box boxes_idx
% associated with this path is removed so that when the next path will be generated it will consider the remaining 
% boxes. e.g. from frame 1 , out of 5 boxes, box no 2 is used to generate path 1, so we
% remove the box from the list, so when we will generate path 2, we will not use this box 2 again


% This version is corrected to keep track of the correct boxes indices in each 
% path , as the boxes are removed after generating each path , to keep track of the indices 
% of the boxes following lines are added :

% index = frames(V(j)).boxes_idx(id);
% index =  [index;  frames(V(j+1)).boxes_idx(id)]; %% newly added
% frames(V(j)).boxes_idx(id)  = []; %% newly added
% ---------------------------------------------------------

% set of vertices
num_frames = length(frames);
V = 1:num_frames;
isempty_vertex = false(length(V),1);

% CHECK: make sure data is not empty
for i=1:num_frames
    if isempty(frames(i).boxes)
        disp(i);
        error('Empty frame');
    end
end


%%--- dymamic programming (K-paths)
path_counter = 0;

while ~any(isempty_vertex)
    
    %fprintf('# of iteration = %d\n',path_counter);
    % step1: initialize data structures
    T = length(V);
    for i=1:T
        num_states = size(frames(V(i)).boxes,1);
        data(i).scores = zeros(num_states,1);
        data(i).index  = nan(num_states,1);
    end
    
    % step2: solve viterbi -- forward pass -- 
    % we do the forward pass from the frame before the last frame (T-1),
    % computing the edge scores between the boxes at last frame (T) and the frame before the last frame (T-1), and continue this till we
    % reach the first frame, 
    % at the first iteration when we compute the edge scores between frames
    % T-1 and frame T and add it with the data(i+1).scores', the
    % data(i+1).scores' values are all 0, this is the case only at first
    % iteration, and at the end of the first iteration data(i+1).scores'
    % are initialised with the max edge scores for each boxes in the frame
    % T-1, in the subsequent iteration, we add the max edge scores of boxes from i+1-th frame to the edge scores of boxes of i-th frame 
    % and then we compute the max_edge_scores for i-th frame and store them in data(i).scores and the indices of the boxes havining max_edge_scores
    % with the i+1 frame boxes are stored in data(i).index
    for i=T-1:-1:1
        edge_score = score_of_edge(frames(V(i)),frames(V(i+1)));        
        edge_score = bsxfun(@plus,edge_score,data(i+1).scores');
        [data(i).scores, data(i).index] = max(edge_score,[],2);        
    end
    
    % step3: decode -- backward pass of Viterbi - back tracing
    path_counter = path_counter+1;
    [s, si] = sort(data(1).scores,'descend'); % sort the max_edge_scores of boxes in decreasing order in the first frame
    id = si(1); % index of the box having maximum edge score in the first frame
    score = data(1).scores(id);
    %index = id;
    index = frames(V(1)).boxes_idx(id);
    boxes = frames(V(1)).boxes(id,:);
    scores = frames(V(1)).scores(id,1);
    for j=1:T-1
        id = data(j).index(id); 
        index =  [index;  frames(V(j+1)).boxes_idx(id)]; %% newly added
        boxes =  [boxes;  frames(V(j+1)).boxes(id,:)];
        scores = [scores; frames(V(j+1)).scores(id,:)];
    end
    paths(path_counter).total_score = score/num_frames;
    paths(path_counter).idx   = index;
    paths(path_counter).boxes = boxes;
    paths(path_counter).scores = scores;
    
    % step4: remove covered boxes
    for j=1:T
        id = paths(path_counter).idx(j);
        id11 = id==frames(V(j)).boxes_idx; % box id 2 remove, because it is included in this path
        id2Rem = find(id11); % % box id 2 remove
        frames(V(j)).boxes(id2Rem,:) = [];
        %frames(V(j)).feat(id2Rem,:)  = [];
        frames(V(j)).scores(id2Rem)  = [];
        frames(V(j)).boxes_idx(id2Rem)  = []; %% newly added
        isempty_vertex(j) = isempty(frames(V(j)).boxes);
    end
    
end

% -------------------------------------------------------------------------
function score = score_of_edge(v1,v2)
% -------------------------------------------------------------------------

N1 = size(v1.boxes,1);
N2 = size(v2.boxes,1);
score = nan(N1,N2);

bounds1 = [v1.boxes(:,1:2) v1.boxes(:,3:4)-v1.boxes(:,1:2)+1];
bounds2 = [v2.boxes(:,1:2) v2.boxes(:,3:4)-v2.boxes(:,1:2)+1];

% f1 = v1.feat;
% f1 = bsxfun(@rdivide,f1,sqrt(sum(f1.^2,2)));
% f2 = v2.feat;
% f2 = bsxfun(@rdivide,f2,sqrt(sum(f2.^2,2)));

for i1=1:N1
    % feature similarity
    feat_similarity = 0;
    %   feat_similarity = f1(i1,:)*f2'; % cosine of angle
    
    % intersectin over union
    iou = inters_union(bounds1(i1,:),bounds2);
    
    % scores
    scores2 = v2.scores;
    scores1 = v1.scores(i1);
    
    score(i1,:) = scores1+scores2'+feat_similarity+iou;
    
end



% -------------------------------------------------------------------------
function iou = inters_union(bounds1,bounds2)
% -------------------------------------------------------------------------

inters = rectint(bounds1,bounds2);
ar1 = bounds1(:,3).*bounds1(:,4);
ar2 = bounds2(:,3).*bounds2(:,4);
union = bsxfun(@plus,ar1,ar2')-inters;

iou = inters./(union+eps);
