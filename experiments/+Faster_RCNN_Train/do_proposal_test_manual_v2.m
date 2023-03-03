function do_proposal_test_manual_v2(conf, model_stage, output_dir, final_test, imdb, det_imgids, type)

if ~final_test
    cache_name = model_stage.cache_name;
else
    cache_name = [model_stage.cache_name '_final'];
end

per_nms_topN = model_stage.nms.per_nms_topN;
nms_overlap_thres = model_stage.nms.nms_overlap_thres;
after_nms_topN =  model_stage.nms.after_nms_topN;
use_gpu = conf.use_gpu;
proposal_test_manual_v2(conf, imdb, output_dir, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu, det_imgids, type, ...
    'net_def_file', model_stage.test_net_def_file, 'net_file', model_stage.output_model_file, 'cache_name', cache_name);

end





