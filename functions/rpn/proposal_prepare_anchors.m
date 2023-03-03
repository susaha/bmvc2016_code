%%--------------------------------------------------------------------------------------------------------------------------------
function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file, output_dir, type)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------
%%--------------------------------------------------------------------------------------------------------------------------------
[output_width_map, output_height_map] = proposal_calc_output_size(conf, test_net_def_file);

anchors  = proposal_generate_anchors(cache_name, output_dir, type, 'scales',  2.^[3:5]);
end

