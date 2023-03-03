% -------------------------------------------------------------------
% Author: Suman Saha
% This script is adapted from Faster R-CNN(*) matlab code repository:
% https://github.com/ShaoqingRen/faster_rcnn
% -------------------------------------------------------------------
function extractFRCNNDTBoxes()

clc;
clear mex;
clear is_valid_handle; % to clear init_key

fileIDc = fopen('../code_base_path.txt','r');
codebase_path = fscanf(fileIDc,'%s');
run([codebase_path '/' 'startup']);

if 0
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = 1; %auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version); % uncomment
end

fileIDb = fopen('../base_dir_path.txt','r');
base_dir_path = fscanf(fileIDb,'%s');

% overidding the basedirpath value for sanity check of the flow models -
% comment this line
%base_dir_path = '/mnt/sun-beta/ss-workspace/bmvc2016_flow_models_sanity_check';



fileIDtm = fopen('../tinyModel.txt','r');
strtm = fscanf(fileIDtm,'%s');
fclose(fileIDtm);
if strcmp(strtm, 'true')
    tinyModel = true;
else
    tinyModel = false;
end
% model
type = 'flow';
model.stage1_rpn.cache_name = 'rpn';
model.stage1_fast_rcnn.cache_name = 'frcnn';
if tinyModel
    model = Model.VGG16_UCF101_tinyModel(model, type, codebase_path);
else
    model = Model.VGG16_UCF101(model, codebase_path);
end

rpn1_boxes_path = [base_dir_path '/' 'rpn_prop_for_frcnn_testing_' type '/aboxes_1'];
output_dir =  [base_dir_path '/' 'fast_rcnn_detection_boxes_' type];
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

im_short_edg_scale = 600;

conf_fast_rcnn  = fast_rcnn_config('scales', im_short_edg_scale, 'test_scales',  im_short_edg_scale, 'image_means', model.mean_image);
%model  = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, cache_base_fast_rcnn, model);

model.stage1_fast_rcnn.output_model_file = [base_dir_path '/trained_models/frcnn1/final'];

actionid = 1:3; % to run in btach mode
Faster_RCNN_Train.do_fast_rcnn_test_ucf101(conf_fast_rcnn, model.stage1_fast_rcnn, output_dir, rpn1_boxes_path, type, actionid);

flcose(fileIDc);
fclose(fileIDb);
end
