function do_proposal_test_ucf101_robocar(conf, model_stage, output_dir, type, actionid)


per_nms_topN = model_stage.nms.per_nms_topN;
nms_overlap_thres = model_stage.nms.nms_overlap_thres;
after_nms_topN =  model_stage.nms.after_nms_topN;
use_gpu = conf.use_gpu;

proposal_test_ucf101_robocar(conf, output_dir, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu, type, actionid,...
    'net_def_file', model_stage.test_net_def_file, 'net_file', model_stage.output_model_file);

end