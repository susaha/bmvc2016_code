% -------------------------------------------------------------------
% Author: Suman Saha
% This script is adapted from Faster R-CNN(*) matlab code repository: 
% https://github.com/ShaoqingRen/faster_rcnn
% -------------------------------------------------------------------
function proposal_test_manual_v2(conf, imdb, output_dir, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu, det_imgids, type, varargin)
%%-- inputs
ip = inputParser;
ip.addRequired('conf',                                      @isstruct);
ip.addRequired('imdb',                                      @isstruct);
ip.addRequired('output_dir',                                @isstr);
ip.addRequired('per_nms_topN',                              @isnumeric);
ip.addRequired('nms_overlap_thres',                         @isnumeric);
ip.addRequired('after_nms_topN',                            @isnumeric);
ip.addRequired('use_gpu',                                   @islogical);
ip.addRequired('det_imgids',                                @isnumeric);
ip.addRequired('type',                                      @isstr);

ip.addParamValue('net_def_file',    fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'test.prototxt'), @isstr);
ip.addParamValue('net_file',        fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'Zeiler_conv5.caffemodel'), @isstr);
ip.addParamValue('cache_name',      'Zeiler_conv5', @isstr);
ip.addParamValue('suffix',          '',             @isstr);
ip.parse(conf, imdb, output_dir, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu, det_imgids, type, varargin{:});
opts = ip.Results;

disp('opts:');
disp(opts);
disp('conf:');
disp(conf);

ab_save_path_1 = [opts.output_dir '/' 'rpn_prop_for_frcnn_training_' type];
if ~exist(ab_save_path_1, 'dir')
    mkdir(ab_save_path_1);
    fprintf('folder created \n %s \n', ab_save_path_1);
    fprintf('press any key...\n');
    %pause;
end

ab_save_path = [ab_save_path_1 '/' 'filtered_aboxes'];
if ~exist(ab_save_path, 'dir')
    mkdir(ab_save_path);
    fprintf('folder created \n %s \n', ab_save_path);
    fprintf('press any key...\n');
    %pause;
end
%%-- init net
% init caffe net

caffe_log_file_base = fullfile(ab_save_path_1, 'caffe_log');
caffe.init_log(caffe_log_file_base);
caffe_net = caffe.Net(opts.net_def_file, 'test');
caffe_net.copy_from(opts.net_file);

% init log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
mkdir_if_missing(fullfile(ab_save_path_1, 'log'));
log_file = fullfile(ab_save_path_1, 'log', ['test_', timestamp, '.txt']);
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

count = 0;

for i = det_imgids %num_images
    ab_save_file = [ab_save_path '/' imdb.image_ids{i} '.mat'];
    if exist(ab_save_file,'file')
        fprintf('aboxes mat file exist:= %s \n', imdb.image_ids{i});
        continue;
    end
    th = tic;
    im = imread(imdb.image_at(i));
    [boxes, scores, ~,~,~] = proposal_im_detect(conf, caffe_net, im);
    fprintf('RPN detection time: %.3fs\n', toc(th));
    aboxes =  [boxes, scores];
    [aboxes, ~]  = boxes_filter_v2(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu);
    save(ab_save_file, 'aboxes');
    fprintf('aboxes for imgid %s saved in:\n %s \n',  imdb.image_ids{i}, ab_save_file);
    if i==det_imgids(1)
        disp('press any key to continue...');
        end
end
diary off;
caffe.reset_all();
rng(prev_rng);

end

%------------------------------------------------------------------------------------------------------------------
function [aboxes, boxes_num] = boxes_filter_v2(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
%------------------------------------------------------------------------------------------------------------------
if per_nms_topN > 0
    aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
end
% do nms
if nms_overlap_thres > 0 && nms_overlap_thres < 1
    if use_gpu
        aboxes = aboxes(nms(aboxes, nms_overlap_thres, use_gpu), :);
    else
        aboxes = aboxes(nms(aboxes, nms_overlap_thres), :);
    end
end
boxes_num = size(aboxes,1);

if after_nms_topN > 0
    aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
end
end
