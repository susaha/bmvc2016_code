% -------------------------------------------------------------------
% Author: Suman Saha & Gurkirt Singh
% gen_action_paths(): first boosts the classification scores of fast r-cnn detection boxes
% using a novel fusion strategy for merging appearance and motion cues based on the
% softmax probability scores and spatial overlaps of the detection bounding boxes.
% This fusion strategy is explained in our BMVC 2016 paper:
% Deep Learning for Detecting Multiple Space-Time Action Tubes in Video.
% Once the scores of detection bounding boxes are boosted, a first pass of
% dynamic programming is applied to construct the action paths within each test video.

% -------------------------------------------------------------------
function gen_action_paths()
close all;
clc;
clear all;
clear mex;
clear is_valid_handle; % to clear init_key

fileIDc = fopen('../code_base_path.txt','r');
codebase_path = fscanf(fileIDc,'%s');
run([codebase_path '/' 'startup']);
fclose(fileIDc);

fileIDb = fopen('../base_dir_path.txt','r');
base_dir_path = fscanf(fileIDb,'%s'); 
fclose(fileIDb);

dt_boxes_path_spatial = [base_dir_path '/' 'fast_rcnn_detection_boxes_spatial'];
dt_boxes_path_flow    = [base_dir_path '/' 'fast_rcnn_detection_boxes_flow'];

save_path = [base_dir_path '/' 'action_paths'];
if ~exist(save_path,'dir')
    mkdir(save_path);
end

fileID = fopen('../spatial_img_path.txt','r');
img_path = fscanf(fileID,'%s');
fclose(fileID);

%load('/mnt/sun-beta/ss-workspace/UCF101_data/annotation/APT_annot/validation/annot.mat'); % videos
st_vid_list = load([base_dir_path '/' 'ucf101_annot' '/testlist01_video_list.mat']); % video
videos = st_vid_list.video;
clear st_vid_list;

num_videos = length(videos);
num_actions = 24;
nms_th = 0.3;
topk_boxes = 5;
%softmax_sc_th = 0; %0.6;

for i=1:num_videos
    action = videos(i).class;
    videoName = videos(i).name;
    
    flist = dir([img_path '/' action '/' videoName '/*.jpg']);
    num_imgs = length(flist);
    frames = struct([]);
    ts = tic;
    
    % make parfor
    parfor f=1:num_imgs % parfor
        fname = flist(f).name;
        filename = replace_file_ext(fname, 'mat');
        
        dt_boxes_file_spatial = [dt_boxes_path_spatial '/'  action '/' videoName  '/' filename];
        [frames(f).boxes_spatial, frames(f).scores_spatial] = read_dt_files(dt_boxes_file_spatial);
        
        dt_boxes_file_flow = [dt_boxes_path_flow '/'  action '/' videoName  '/' filename];
        [frames(f).boxes_flow, frames(f).scores_flow] = read_dt_files(dt_boxes_file_flow);
    end
    
    action_paths = cell(num_actions,1);
    
    %make parfor
    parfor a = 1:num_actions    % parfor    
        action_paths{a,1} = get_action_paths(frames, a, nms_th, num_actions, topk_boxes);
    end
    te =toc(ts);
    save_file = [save_path '/' videoName '.mat'];
    save(save_file,'action_paths');
    fprintf('action paths saved in %.3fs at location: \n %s \n', te, save_file);
end
end


function paths = get_action_paths(frames, a, nms_th, num_actions, topk_boxes)
a = a+1; % to skip background boxes and scores
action_frames = struct([]);

for f = 1:length(frames)
    %%------ SPATIAL-----------
    boxes_spatial = frames(f).boxes_spatial;
    scores_spatial = frames(f).scores_spatial;
    boxes_spatial = [boxes_spatial(:, (1+(a-1)*4):(a*4)), scores_spatial(:, a)];
    pick_nms_spatial = nms(boxes_spatial, nms_th);    
    pick_nms_spatial = pick_nms_spatial(1:topk_boxes);    
    boxes_spatial = boxes_spatial(pick_nms_spatial, :); 
      
    
    %%------ MOTION-----------
    boxes_flow = frames(f).boxes_flow;
    scores_flow = frames(f).scores_flow;
    boxes_flow = [boxes_flow(:, (1+(a-1)*4):(a*4)), scores_flow(:, a)];
    pick_nms_flow = nms(boxes_flow, nms_th);    
    pick_nms_flow = pick_nms_flow(1:topk_boxes);    
    boxes_flow = boxes_flow(pick_nms_flow, :); 
    
     %%---- BOOST THE SCORE OF SPATIAL BOXES AS PER THE IOU BETWEEN FLOW AND
    %%---- SPATIAL BOXES AND THE SOFTMAX SCORE OF FLOW BOX -----    
    [boxes_spatial, ~] = boost_boxes_spatial(boxes_spatial, boxes_flow);
    
    if 0
        boxes = cat(1, boxes_spatial, boxes_flow);
        scores = cat(1, scores_spatial, scores_flow);
        pick_nms = cat(1, pick_nms_spatial, pick_nms_flow);
    end
    
    boxes = boxes_spatial;
    scores = scores_spatial;
    pick_nms = pick_nms_spatial;
    
    action_frames(f).boxes =  boxes(:, 1:4);
    action_frames(f).scores =  boxes(:, 5);
    % using the sotmax score for action a --> boxes(:, 5) and the
    % 1-boxes(:, 5) tells us the probability of the action not happeing 
    action_frames(f).allScores = [ boxes(:, 5) 1-boxes(:, 5) scores(pick_nms,1)]; % not using the 3rd column:scores(pick_nms,1) --> background class region lprop scores 
    action_frames(f).boxes_idx = 1:size(action_frames(f).boxes,1); 
end
clear boxes scores pick_nms pick_softmax;

%%---------- zero_jump_link ------------
paths = zero_jump_link(action_frames);
%%--------------------------------------

path_all_score = zeros(length(frames),3); % 3 is for : (action softmax score ; 1 - action's softmax score; background boxes scores)
for p = 1 : length(paths)
    for f = 1:length(frames)
        path_all_score(f,:) = action_frames(f).allScores(paths(p).idx(f),:);
    end
    paths(p).allScore = path_all_score;
end

end

function [bs, bf] = boost_boxes_spatial(bs, bf) % bs - boxes_spatial bf-boxes_flow
nb = size(bs,1); % num boxes
iou_thr = 0.3;

box_spatial = [bs(:,1:2) bs(:,3:4)-bs(:,1:2)+1];
box_flow =    [bf(:,1:2) bf(:,3:4)-bf(:,1:2)+1];


for i=1:nb
    ovlp = inters_union(box_spatial(i,:), box_flow); % ovlp has 1x5 or 5x1 dim
    [movlp, mind] = max(ovlp);
    if movlp>=iou_thr;
        bs(i,5) = bs(i,5) + bf(mind,5)*movlp;
        % bs(i,5) = bs(i,5) + bf(mind,5)*movlp - abs(bs(i,5)-bf(mind,5));
        %bf(mind,5) = bf(mind,5) + bs(i,5)*movlp;
    end
end

end

% ------------------------------------------------------------------------
function iou = inters_union(bounds1,bounds2)
% ------------------------------------------------------------------------
inters = rectint(bounds1,bounds2);
ar1 = bounds1(:,3).*bounds1(:,4);
ar2 = bounds2(:,3).*bounds2(:,4);
union = bsxfun(@plus,ar1,ar2')-inters;
iou = inters./(union+0.001);
end


function [boxes, scores] = read_dt_files(dt_boxes_file)
st = load(dt_boxes_file);
boxes = st.boxes;
scores =  st.scores;
clear st;
end

function filename = replace_file_ext(filename, ext)
ind = strfind(filename, '.');
str = filename(1:ind-1);
filename = [str '.' ext];
end


