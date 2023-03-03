clc;
clear all;
close all;

fileIDb = fopen('../base_dir_path.txt','r');
base_dir_path = fscanf(fileIDb,'%s');
fclose(fileIDb);
tube_path = [base_dir_path '/action_tubes'];
action_tubes = [tube_path '/' 'tubes.mat'];
st = load(action_tubes);
tubes = st.tubes; clear st;

% loadining the UCF101 testlist01 annotation
 st_annot = load([base_dir_path '/ucf101_annot/annot.mat']); % videos - new modifed APT annot 
 videos = st_annot.videos; clear st_annot;
 
% minimum  number of frames required to construct a valid action tube
min_num_frames = 30; 
topk = 40; % this is not used in this evaluation
xmldata = tube2Xml(tubes, min_num_frames, topk, videos);
save_path = [base_dir_path '/run_evaluation'];
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
save([save_path '/' 'xmldata.mat'], 'xmldata');
