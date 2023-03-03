% -------------------------------------------------------------------
% Author: Suman Saha
% This script copies spatial/flow images to a specific folder for
% RPN and F-RCNN training.
% Training and validation images are numbered sequentially,
% e.g. among 100 total images first 70 training images are numbered
% from 00001.jpg,..., 00070.jpg and remaining 30 validation images
% are numbered 00071.jpg,..., 00100.jpg.
% Also it saves the following mat files under folder 'output/imdb_roidb_cache':
% 1> ucf101_annotation_trainsplit01.mat
% 2> ucf101_annotation_validation-split01.mat
% 3> ucf101_trainsplit01_imdb_inputs.mat
% 4> ucf101_validation-split01_imdb_inputs.mat
% which are used to generate training and validation data.
% -------------------------------------------------------------------
function prep_data(video, annot, trainval_video_ids, target, save_path, split, img_path, img_save_path, img_id, tinyModel)
actions = {'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing',...
    'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', ...
    'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'};

if tinyModel
    num_classes = 2;
else
    num_classes = length(actions);
end

assert(length(trainval_video_ids) == length(actions));
num_gt_instances = 0;
ant_ind = 0;
fid = fopen([save_path '/' 'bad_gt_boxes_log.txt'],'w');
img_w = 320;
img_h = 240;
for a=1:num_classes
    assert(strcmp(trainval_video_ids{a,3}, actions{a}) );
    trainset = trainval_video_ids{a,target};
    
    % this is for a tinyModel where in training set we have 2 videos and in
    % validation set we have 1 video for each of 2 action classes
    if tinyModel
        if target==1
            trainset = trainset(1:2,:);
        else
            trainset = trainset(1,:);
        end
    end
    
    for i=trainset'
        videoName = video(i).name; 
        action = video(i).class;        
        fprintf('%s %s\n', action, videoName);
        frm_ind = find(strcmp(videoName, annot.videoName));
        num_frames = length(frm_ind);
        action_id = find(strcmp(action, actions));  
        for j=1:num_frames
            ind = frm_ind(j);
            action =  cell2mat(annot.action(ind));
            framenr = cell2mat(annot.framenr(ind));
            gt_boxes_ = cell2mat(annot.boxes(ind));  % are in [ x y w h]            
            if ~isempty(gt_boxes_)                
                %%--- converting gt boxes from [x y w h] to [x1 y1 x2 y2]
                gt_boxes_ = [gt_boxes_(:,1:2) gt_boxes_(:,1:2) + gt_boxes_(:,3:4)];
                type = 0;
                gt_boxes_f = filterBoxes(gt_boxes_, img_w, img_h, fid, [ action '/' videoName], type);                
                if ~isempty(gt_boxes_f)
                    % convert gt_boxes_f back to to x y w h
                    gt_boxes_f = [gt_boxes_f(:,1:2) gt_boxes_f(:,3:4) - gt_boxes_f(:,1:2)];
                    num_gt_instances = num_gt_instances+size(gt_boxes_f,1);
                    img_id = img_id+1;
                    ant_ind = ant_ind+1;                    
                    src_img = sprintf('%s/%s/%s/%05d.jpg', img_path, action, videoName, framenr);
                    dest_img = sprintf('%s/%06d.jpg', img_save_path, img_id);  
                    str_cmd = [ 'cp ' src_img ' ' dest_img];
                    system(str_cmd); % uncomment
                    fprintf('copying file:  %s \n', str_cmd);                    
                    %%------- GENERATING ANNOTATION FOR roidb_from_ucf101.m -----------
                    if 1
                        annotation{ant_ind, 1} = gt_boxes_f;
                        annotation{ant_ind, 2} = sprintf('%06d',img_id);
                        annotation{ant_ind, 3} = action_id;
                    end                    
                end
            end
        end
    end
end

imdb_inputs.num_gt_instances = num_gt_instances;
imdb_inputs.img_id = img_id;
save([save_path '/' 'ucf101_' split '_imdb_inputs.mat'], 'imdb_inputs');
save([save_path '/' 'ucf101_annotation_' split '.mat'], 'annotation'); 
fclose(fid);
end

% -------------------------------------------------------------------
% takes boxes and filter them and put the x1 y1 x2 y2 values
% with in range [1 1 320 240] 
% -------------------------------------------------------------------
function [out_boxes] = filterBoxes(inp_boxes, img_w, img_h, fid, act_vid, type)
if type == 1
    box_type = 'SS';
else
    box_type = 'GT';
end
out_boxes = [];
num_boxes = size(inp_boxes,1);
for i=1:num_boxes
    box = inp_boxes(i,:);
    x1 = box(1);
    y1 = box(2);
    x2 = box(3);
    y2 = box(4);    
    %### some ground truth boxes have x1 and y1 negative values
    %### some SS boxes have bounds outside of the image size range
    %### the following lines will deal with those instances
    x1_ = max(x1,1); % if the x1 value is < 1 then x1_ = 1
    y1_ = max(y1,1); % if the y1 value is < 1 then y1_ = 1
    x2_ = min(x2,img_w); % if the x2 value is > 320 then x2_ = 320
    y2_ = min(y2,img_h); % if the y2 value is > 240 then y2_ = 240    
    %### this is the final sanity check of all boxes wehther they have a
    %valid bounds
    if x1_ < x2_ && y1_ < y2_
        box_ = [x1_ y1_ x2_ y2_];
        out_boxes = cat(1, out_boxes, box_);
    else
        fprintf(fid, 'Box type: %s\n', box_type);
        fprintf(fid, '%s\n', act_vid);
        fprintf(fid, 'box has wrong coordinates: x1=%d y1=%d x2=%d y2=%d\n', x1, y1, x2, y2);
        fprintf(fid, 'box has wrong coordinates: x1_=%d y1_=%d x2_=%d y2_=%d\n', x1_, y1_, x2_, y2_);
    end
end
end