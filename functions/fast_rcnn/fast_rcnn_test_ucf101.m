% -------------------------------------------------------------------
% Author: Suman Saha
% This script is adapted from Faster R-CNN(*) matlab code repository: 
% https://github.com/ShaoqingRen/faster_rcnn
% -------------------------------------------------------------------
function fast_rcnn_test_ucf101(conf, output_dir, rpn1_boxes_path, type, actionid, varargin)  % imdb, roidb,


%%--- inputs
ip = inputParser;
ip.addRequired('conf',                                          @isstruct);
ip.addRequired('output_dir',                                    @isstr);
ip.addRequired('rpn1_boxes_path',                               @isstr);
ip.addRequired('type',                               @isstr);
ip.addRequired('actionid',                               @isnumeric);

ip.addParamValue('net_def_file',    '', 			@isstr);
ip.addParamValue('net_file',        '', 			@isstr);
ip.addParamValue('cache_name',      '', 			@isstr);
ip.addParamValue('suffix',          '',             @isstr);
ip.addParamValue('ignore_cache',    false,          @islogical);

ip.parse(conf, output_dir, rpn1_boxes_path, type, actionid, varargin{:});
opts = ip.Results;

%--- UCF TESTLIST01 ANNOTATION AND IMAGE PATH -----
fileID = fopen('../base_dir_path.txt','r');
base_dir_path = fscanf(fileID,'%s'); 

% overidding the basedirpath value for sanity check of the flow models -
% comment this line 
base_dir_path = '/mnt/sun-beta/ss-workspace/bmvc2016_flow_models_sanity_check';

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


%%---  set cache dir
save_path = opts.output_dir; 

rpn1_boxes_path = opts.rpn1_boxes_path;

caffe_log = [save_path '/' 'frcnn1_caffe_log'];
mkdir_if_missing(caffe_log);
%%---  init log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
mkdir_if_missing(fullfile(caffe_log, 'log'));
log_file = fullfile(caffe_log, 'log', ['test_', timestamp, '.txt']);
diary(log_file);

% init caffe net
caffe_log_file_base = fullfile(caffe_log, 'caffe_log');
caffe.init_log(caffe_log_file_base);
%--- fast rcnn net
caffe_net = caffe.Net(opts.net_def_file, 'test');
caffe_net.copy_from(opts.net_file);

% set random seed
prev_rng = seed_rand(conf.rng_seed);
caffe.set_random_seed(conf.rng_seed);

% set gpu/cpu
if conf.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end

% determine the maximum number of rois in testing
%max_rois_num_in_gpu = check_gpu_memory(conf, caffe_net);

disp('opts:');
disp(opts);
disp('conf:');
disp(conf);

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
                
                rpn1_boxes_file = sprintf('%s/%s/%s/%05d.mat', rpn1_boxes_path, action, videoName, j);
                
                if ~exist(rpn1_boxes_file, 'file')
                    fprintf('file not exist = \n %s \n', rpn1_boxes_file);
                    pause;
                end
                abst = load(rpn1_boxes_file);
                aboxes = abst.aboxes; clear abst;
                
                %---- FRCNN DETECTION----
                th = tic();
                [boxes, scores] = fast_rcnn_im_detect(conf, caffe_net, im,  aboxes(:, 1:4), after_nms_topN);
                t_detection = toc(th);
                fprintf('frame: %s \n : detection time %.3fs \n', [action '/' videoName], t_detection);
                
                save(save_file, 'boxes', 'scores');  
                fprintf('file saved in \n %s \n', save_file);
                
                if i==1 &&  j==1
                    fprintf('press any key to continue...\n');
                    pause;
                end
%                 dt_boxes = [boxes(:, (1+(a-1)*4):(a*4)), scores(:, a)]; % [x1 y1 x2 y2 class specific score]
%                 save(save_file, 'dt_boxes');
               % pause;
            end
        end
    end
end


if 0
    for i = 1:num_images
        im = imread(imdb.image_at(i));
        %--- rpn generatining region prop/aboxes ----
        
        
        %--- fast rcnn detection taking boxes from rpn ----
        [boxes, scores] = fast_rcnn_im_detect(conf, caffe_net, im,  aboxes(:, 1:4), max_rois_num_in_gpu);
    end
    
end

caffe.reset_all();
rng(prev_rng);
%end
diary off;
end



function max_rois_num = check_gpu_memory(conf, caffe_net)
%%---  try to determine the maximum number of rois

max_rois_num = 0;
for rois_num = 500:500:5000
    % generate pseudo testing data with max size
    im_blob = single(zeros(conf.max_size, conf.max_size, 3, 1));
    rois_blob = single(repmat([0; 0; 0; conf.max_size-1; conf.max_size-1], 1, rois_num));
    rois_blob = permute(rois_blob, [3, 4, 1, 2]);
    
    net_inputs = {im_blob, rois_blob};
    
    % Reshape net's input blobs
    caffe_net.reshape_as_input(net_inputs);
    
    caffe_net.forward(net_inputs);
    gpuInfo = gpuDevice();
    
    max_rois_num = rois_num;
    
    if gpuInfo.FreeMemory < 2 * 10^9  % 2GB for safety
        break;
    end
end

end

