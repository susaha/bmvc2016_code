% -------------------------------------------------------------------
% Author: Suman Saha
% This script is adapted from Faster R-CNN(*) matlab code repository: 
% https://github.com/ShaoqingRen/faster_rcnn
% -------------------------------------------------------------------
function model = VGG16_UCF101(model, codebase_path)
% function model = VGG16_UCF101(model, type, codebase_path)
% VGG 16layers (only finetuned from conv3_1)

% fileID = fopen('../../code_base_path.txt','r');
% codebase_path = fscanf(fileID,'%s'); 
% fclose(fileID);
% if strcmp(type,'spatial')
%     model.mean_image                                = [codebase_path '/' 'models/pre_trained_models/mean_image.mat'];
% else
%     model.mean_image                                = [codebase_path '/' 'models/pre_trained_models/mean_image_' type '.mat'];
% end

 model.mean_image                                = [codebase_path '/' 'models/pre_trained_models/mean_image.mat'];
 
model.pre_trained_net_file                          = [codebase_path '/' 'models/pre_trained_models/vgg16.caffemodel']; 

% Stride in input image pixels at the last conv layer
model.feat_stride                                   = 16;

%%- rpn, 
model.stage1_rpn.solver_def_file                    = [codebase_path '/' 'models/rpn_prototxts/solver_240k320k.prototxt']; 
model.stage1_rpn.test_net_def_file                  = [codebase_path '/' 'models/rpn_prototxts/test.prototxt']; 
model.stage1_rpn.init_net_file                      = model.pre_trained_net_file;

% rpn test setting
model.stage1_rpn.nms.per_nms_topN                   = -1;
model.stage1_rpn.nms.nms_overlap_thres              = 0.7;
model.stage1_rpn.nms.after_nms_topN                 = 2000;

%%- fast rcnn, pre-trained network
model.stage1_fast_rcnn.solver_def_file              = [codebase_path '/' 'models/fast_rcnn_prototxts/solver_240k320k.prototxt']; 
model.stage1_fast_rcnn.test_net_def_file            = [codebase_path '/' 'models/fast_rcnn_prototxts/test.prototxt'];
model.stage1_fast_rcnn.init_net_file                = model.pre_trained_net_file;


% rpn test setting
model.stage2_rpn.nms.per_nms_topN                   = -1;
model.stage2_rpn.nms.nms_overlap_thres              = 0.7;
model.stage2_rpn.nms.after_nms_topN                 = 2000;

%%- final test
model.final_test.nms.per_nms_topN                   = 6000; % to speed up nms
model.final_test.nms.nms_overlap_thres              = 0.7;
model.final_test.nms.after_nms_topN                 = 300;


end