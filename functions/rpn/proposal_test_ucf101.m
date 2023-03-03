function proposal_test_ucf101(conf, output_dir, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu, type, actionid,  varargin)

%%-- inputs
ip = inputParser;
ip.addRequired('conf',                                      @isstruct);
ip.addRequired('output_dir',                                @isstr);
ip.addRequired('per_nms_topN',                              @isnumeric);
ip.addRequired('nms_overlap_thres',                         @isnumeric);
ip.addRequired('after_nms_topN',                            @isnumeric);
ip.addRequired('use_gpu',                                   @islogical);
ip.addRequired('type',                                      @isstr);
ip.addRequired('actionid',                                  @isnumeric);


ip.addParamValue('net_def_file',    fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'test.prototxt'), @isstr);
ip.addParamValue('net_file',        fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'Zeiler_conv5.caffemodel'), @isstr);
ip.addParamValue('cache_name',      'Zeiler_conv5', @isstr);
ip.addParamValue('suffix',          '',             @isstr);
ip.parse(conf, output_dir, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu, type, actionid, varargin{:});
opts = ip.Results;

disp('opts:');
disp(opts);
disp('conf:');
disp(conf);

%--- UCF TESTLIST01 ANNOTATION AND IMAGE PATH -----
fileID = fopen('../base_dir_path.txt','r');
base_dir_path = fscanf(fileID,'%s');
fclose(fileID);

% overidding the basedirpath value for sanity check of the flow models -
% comment this line 
%base_dir_path = '/mnt/sun-beta/ss-workspace/bmvc2016_flow_models_sanity_check';

annot_path = [base_dir_path '/' 'ucf101_annot'];
load([annot_path '/' 'testlist01_video_list.mat']);

% reading image path
if strcmp(type,'spatial')
    fileID = fopen('../spatial_img_path.txt','r');
    img_path = fscanf(fileID,'%s');
    fclose(fileID);
else
    fileID = fopen('../flow_img_path.txt','r');
    img_path = fscanf(fileID,'%s');
    fclose(fileID);
end


actions = {'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing',...
    'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin', ...
    'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog'};


cache_dir = opts.output_dir; 

save_path = [cache_dir '/' 'aboxes'];
if ~exist(save_path, 'dir')
    mkdir(save_path);
    fprintf('folder created \n %s', save_path);
end

caffe_log = [cache_dir '/' 'rpn1_caffe_log'];

%%-- init net
% init caffe net
mkdir_if_missing(caffe_log);
caffe_log_file_base = fullfile(caffe_log, 'caffe_log');
caffe.init_log(caffe_log_file_base);
caffe_net = caffe.Net(opts.net_def_file, 'test');
caffe_net.copy_from(opts.net_file);

% init log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
mkdir_if_missing(fullfile(caffe_log, 'log'));
log_file = fullfile(caffe_log, 'log', ['test_', timestamp, '.txt']);
diary(log_file);

% set random seed
prev_rng = seed_rand(conf.rng_seed);
caffe.set_random_seed(conf.rng_seed);

% set gpu/cpu
if conf.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end

% disp('opts:');
% disp(opts);
% disp('conf:');
% disp(conf);


after_nms_topN = 300; 

act = actionid;
for a = act
  
    for i=1:length(video)
        action = video(i).class;
        videoName = video(i).name;
        
        if strcmp(actions{a}, action)
            fprintf('%s\n', videoName);
            img_list = dir([img_path '/' action '/' videoName '/*.jpg']);
            num_imgs = length(img_list);
            
            for j = 1:num_imgs
                img_file = sprintf('%s/%s/%s/%05d.jpg', img_path, action, videoName, j);
                im = imread(img_file);
                
                save_file = sprintf('%s/%s/%s/%05d.mat', save_path, action, videoName, j);
              
                if ~exist([save_path '/' action '/' videoName],'dir')
                    mkdir([save_path '/' action '/' videoName]);
                end
                
                if exist(save_file,'file')
                    fprintf('res present for %s\n', save_file);
                    continue;
                end
                
                th = tic();
                [boxes, scores, ~,~,~] = proposal_im_detect(conf, caffe_net, im);
                aboxes  = boxes_filter([boxes, scores], per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu);
                t_detection = toc(th);
                fprintf('frame: %s \n : detection time %.3fs \n', [action '/' videoName], t_detection);
                save(save_file, 'aboxes');
            end
        end
    end
end


diary off;
caffe.reset_all();
rng(prev_rng);

end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
% to speed up nms
if per_nms_topN > 0
    aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
end
% do nms
if nms_overlap_thres > 0 && nms_overlap_thres < 1
    aboxes = aboxes(nms(aboxes, nms_overlap_thres, use_gpu), :);
end
if after_nms_topN > 0
    aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
end
end

% %------------------------------------------------------------------------------------------------------------------
% function [aboxes, boxes_num] = boxes_filter_v2(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
% %------------------------------------------------------------------------------------------------------------------
% if per_nms_topN > 0
%     aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
% end
% % do nms
% if nms_overlap_thres > 0 && nms_overlap_thres < 1
%     if use_gpu
%         aboxes = aboxes(nms(aboxes, nms_overlap_thres, use_gpu), :);
%     else
%         aboxes = aboxes(nms(aboxes, nms_overlap_thres), :);
%     end
% end
% boxes_num = size(aboxes,1);
% 
% if after_nms_topN > 0
%     aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
% end
% end
