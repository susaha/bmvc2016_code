% -------------------------------------------------------------------
% Author: Suman Saha
% This script generates class wise video ids for UCF-101 train split 1
% -------------------------------------------------------------------
clear all;
close all;
clc;
fileID = fopen('../base_dir_path.txt','r');
base_dir_path = fscanf(fileID,'%s');
fclose(fileID);
annot_path = [base_dir_path '/' 'ucf101_annot'];
load([annot_path '/' 'trainlist01_actions_vs_videos.mat']);
actions = {'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing',...
    'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', ...
    'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'};
class_names = cell(length(video),1);
for i=1:length(video)
    class_names{i,1} = video(i).class;
end
class_video_ids = cell(length(actions),1);
for i=1:length(actions)
    vind = find(strcmp(actions{i}, class_names));
    class_video_ids{i,1} = vind;
    class_video_ids{i,2} = actions{i};
end
save([annot_path '/' 'class_wise_video_ids.mat'], 'class_video_ids');
