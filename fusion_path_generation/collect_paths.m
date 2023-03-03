% -------------------------------------------------------------------
% Author: Suman Saha
% -------------------------------------------------------------------
fileIDb = fopen('../base_dir_path.txt','r');
base_dir_path = fscanf(fileIDb,'%s'); 
fclose(fileIDb);
action_path_cache_files = [base_dir_path '/' 'action_paths'];
st_vid_list =  load([base_dir_path '/' 'ucf101_annot' '/testlist01_video_list.mat']); % video
videos = st_vid_list.video;
clear st_vid_list;
num_videos = length(videos);

for i=1:num_videos
    action = videos(i).class;
    videoName = videos(i).name;
    
    action_path_file = [action_path_cache_files '/' videoName '.mat'];
    st = load(action_path_file);
    video_paths = st.action_paths;
    clear st;
    paths(i).video_id = videoName;
    paths(i).paths = video_paths;
    fprintf('done vid=%d\n', i);
end
save_path = [base_dir_path '/action_paths_final'];
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
save_file = [save_path '/' 'paths.mat'];
save(save_file, 'paths');
fprintf('file saved in %s\n', save_file);