function do_fast_rcnn_test_ucf101(conf, model_stage, output_dir,  rpn1_boxes_path, type, actionid)                                  
     fast_rcnn_test_ucf101(conf, output_dir, rpn1_boxes_path, type, actionid, 'net_def_file',  model_stage.test_net_def_file, 'net_file', model_stage.output_model_file);
end
