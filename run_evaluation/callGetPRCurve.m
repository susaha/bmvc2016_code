actions = {'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing',...
    'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', ...
    'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'};
fileIDb = fopen('../base_dir_path.txt','r');
base_dir_path = fscanf(fileIDb,'%s');
fclose(fileIDb);
xmld_path = [base_dir_path '/run_evaluation'];
xmld_file = [xmld_path '/' 'xmldata.mat'];
st= load();
xmldata = st.xmldata; clear st;
% loadining the UCF101 testlist01 annotation
st_annot = load([base_dir_path '/ucf101_annot/annot.mat']); % videos - new modifed APT annot
videos = st_annot.videos; clear st_annot;
iou_th = 0.2;
% do the main evaluation
[mAP,mAIoU, AP] = get_PR_curve(videos, xmldata, actions, iou_th);
fprintf('mean Average Precision:= %f at spatio-temporal IoU threshold:=%f\n', mAP, iou_th);