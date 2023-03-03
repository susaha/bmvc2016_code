% -------------------------------------------------------------------
% Author: Suman Saha
% This script copies spatial/flow images to a specific folder for
% RPN and F-RCNN training.
% Training and validation images are numbered sequentially,
% e.g. among 100 total images first 70 training images are numbered
% from 00001.jpg,..., 00070.jpg and remaining 30 validation images
% are numbered 00071.jpg,..., 00100.jpg.
% Also it saves the following mat files under folder [base_dir_path '/' 'imdb_roidb_cache']:
% 1> ucf101_annotation_trainsplit01.mat
% 2> ucf101_annotation_validation-split01.mat
% 3> ucf101_trainsplit01_imdb_inputs.mat
% 4> ucf101_validation-split01_imdb_inputs.mat
% which are used to generate training and validation data.
% Set target 1 for training and 2 for validation dataset.
% Set type "spatial" for RGB images and "flow" for optical flow images.
% Run this script twice, first run with target=1 and then target=2.
% Replace "img_path" value to your machine's local path where UCF-101
% RGB/flow images are stored.
% UCF-101 RGB and flow images are to be stored in the following format:
% <action_name>/<video_name>/<image_file_name>.jpg,
% e.g., Basketball/v_Basketball_g01_c01/00001.jpg
% -------------------------------------------------------------------
fileID = fopen('../base_dir_path.txt','r');
base_dir_path = fscanf(fileID,'%s');
fclose(fileID);
annot_path = [base_dir_path '/' 'ucf101_annot'];
load([annot_path '/' 'trainlist01_actions_vs_videos.mat']);
load([annot_path '/' 'annot_train_test_list01_v5.mat']); % annot
load([annot_path '/' 'trainval_video_ids.mat']);
%------------ MAIN INPUT PARAM ---------------
% pepare data for a tiny model with 2 videos in train set and 1 video in validation set
% for each action class where there are 2 action classes in total
fileIDtm = fopen('../tinyModel.txt','r');
strtm = fscanf(fileIDtm,'%s');
fclose(fileIDtm);
if strcmp(strtm, 'true')
    tinyModel = true;
else
    tinyModel = false;
end

target = 2; % 1 for trainset ; 2 for validation set
type = 'flow'; % or flow
save_path = [base_dir_path '/' 'imdb_roidb_cache_' type];
if ~exist(save_path,'dir')
    mkdir(save_path);
end
if target==1
    split = 'trainsplit01'; % or testsplit01
    img_id = 0;
else
    split = 'validation-split01'; % or testsplit01
    train_imi = load([base_dir_path '/' 'imdb_roidb_cache_' type '/' 'ucf101_trainsplit01_imdb_inputs.mat']); % imdb_inputs
    img_id = train_imi.imdb_inputs.img_id;
end
if strcmp(type,'spatial')
    fileID = fopen('../spatial_img_path.txt','r');
    img_path = fscanf(fileID,'%s');
    fclose(fileID);
    img_save_path = [base_dir_path '/' 'frcnn_input_imgs_' type];
    if ~exist(img_save_path,'dir')
        mkdir(img_save_path);
    end
    
elseif strcmp(type,'flow')
    fileID = fopen('../flow_img_path.txt','r');
    img_path = fscanf(fileID,'%s');
    fclose(fileID);
    img_save_path = [base_dir_path '/' 'frcnn_input_imgs_' type];
    if ~exist(img_save_path,'dir')
        mkdir(img_save_path);
    end
end

prep_data(video, annot, trainval_video_ids, target, save_path, split, img_path, img_save_path, img_id, tinyModel);
