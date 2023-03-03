% -------------------------------------------------------------------
% Author: Suman Saha
% This script splits the UCF-101 trainsplit01 in training and validation sets
% -------------------------------------------------------------------
clear all;
close all;
clc;
fileID = fopen('../base_dir_path.txt','r');
base_dir_path = fscanf(fileID,'%s'); 
annot_path = [base_dir_path '/' 'ucf101_annot'];
load([annot_path '/' 'class_wise_video_ids.mat']);
actions = {'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing',...
    'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', ...
    'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'};
trainval_video_ids = cell(length(actions),1);
for i=1:length(actions)
    vids = class_video_ids{i,1};
    assert(strcmp(actions{i}, class_video_ids{i,2}));    
    num_vids = length(vids);
    num_train_vids = round(num_vids*0.7);
    num_test_vids = num_vids - num_train_vids;    
    rand_ind = randperm(num_vids);
    vids = vids(rand_ind);
    trainset = vids(1:num_train_vids);
    valset = vids(num_train_vids+1:end);
    trainval_video_ids{i,1} = trainset;
    trainval_video_ids{i,2} = valset;
    trainval_video_ids{i,3} = actions{i};
end
save([annot_path '/' 'trainval_video_ids.mat'], 'trainval_video_ids');
fclose(fileID);