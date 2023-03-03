% -------------------------------------------------------------------
% Author: Suman Saha
% This script is adapted from Faster R-CNN(*) matlab code repository:
% https://github.com/ShaoqingRen/faster_rcnn
% -------------------------------------------------------------------
function extractRPNProposals()
clc;
clear mex;
clear is_valid_handle; % to clear init_key

fileIDc = fopen('../code_base_path.txt','r');
codebase_path = fscanf(fileIDc,'%s');
run([codebase_path '/' 'startup']);

fileIDb = fopen('../base_dir_path.txt','r');
base_dir_path = fscanf(fileIDb,'%s');

% overidding the basedirpath value for sanity check of the flow models -
% comment this line
%base_dir_path = '/mnt/sun-beta/ss-workspace/bmvc2016_flow_models_sanity_check';
conf_proposal_file = [base_dir_path '/' 'conf_proposal.mat'];

if 1
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = 1; %auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version); % uncomment
end

conf_proposal_path = [base_dir_path '/' 'conf_proposal'];
conf_proposal_file = [conf_proposal_path '/' 'conf_proposal.mat'];
% conf_proposal_st = load(conf_proposal_file);
% conf_proposal = conf_proposal_st.conf_proposal;
% clear conf_proposal_st;

type = 'flow';
fileIDtm = fopen('../tinyModel.txt','r');
strtm = fscanf(fileIDtm,'%s');
fclose(fileIDtm);
if strcmp(strtm, 'true')
    tinyModel = true;
else
    tinyModel = false;
end
% model
model.stage1_rpn.cache_name = 'rpn';
model.stage1_fast_rcnn.cache_name = 'frcnn';
if tinyModel
    model = Model.VGG16_UCF101_tinyModel(model, type, codebase_path);
else
    model = Model.VGG16_UCF101(model, codebase_path);
end

im_short_edg_scale = 600;


if exist(conf_proposal_file,'file')
    st = load(conf_proposal_file);
    conf_proposal = st.conf_proposal;
    clear st;
else
    conf_proposal  = proposal_config('scales', im_short_edg_scale, 'test_scales',  im_short_edg_scale, 'image_means', model.mean_image, 'feat_stride', model.feat_stride);
    % generate anchors and pre-calculate output size of rpn network
    [conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
        = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file, base_dir_path, type);
    save(conf_proposal_file, 'conf_proposal');
end

fprintf('Faster_RCNN_Train.set_cache_folder done\n');

%model.stage1_rpn.output_model_file = [base_dir_path '/trained_models/rpn1/final'];
model.stage1_rpn.output_model_file = [base_dir_path '/rpn_trained_model_spatial/ucf101_trainsplit01/iter_100'];

output_dir = [base_dir_path '/' 'rpn_prop_for_frcnn_testing_' type];
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

actionid = 1:3;
Faster_RCNN_Train.do_proposal_test_ucf101(conf_proposal, model.stage1_rpn, output_dir, type, actionid);

%Faster_RCNN_Train.do_proposal_test_ucf101_robocar(conf_proposal, model.stage1_rpn, output_dir, type, actionid);


fclose(fileIDc);
fclose(fileIDb);
end
