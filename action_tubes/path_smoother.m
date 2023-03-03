% -------------------------------------------------------------------
% Author: Gurkirt Singh
% It applies a second pass dynamic programming on the action paths for 
% temporal label smoothing and trimming to generate the final action tubes.
% -------------------------------------------------------------------
clc;
clear all;
close all;

fileIDb = fopen('../base_dir_path.txt','r');
base_dir_path = fscanf(fileIDb,'%s');
fclose(fileIDb);
action_paths = [base_dir_path '/action_paths_final/paths.mat'];
stp = load(action_paths);
paths = stp.paths;
clear stp;
% loading the class specific alphas optimised on validation set
st = load('max_mAP_class.mat'); % tubes
alpha = st.max_mAP_class(:,2)';
clear st;
num_action = 3; %24; % 2 for tiny model
tubes = par_path_smoother(paths, alpha, num_action);
save_path = [base_dir_path '/action_tubes'];
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
save([save_path '/' 'tubes.mat'], 'tubes');