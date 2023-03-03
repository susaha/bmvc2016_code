% -------------------------------------------------------------------
% Author: Suman Saha
% Computes and caches the bounding box mean, bounding box standard deviation and image_roidb
% This script is adapted from Faster R-CNN(*) matlab code repository: 
% https://github.com/ShaoqingRen/faster_rcnn
% -------------------------------------------------------------------
function  proposal_prepare_image_roidb_v2(conf, imdbs, roidbs, test, output_dir, mode, type, bbox_means, bbox_stds)

if ~exist('bbox_means', 'var')
    bbox_means = [];
    bbox_stds = [];
end
if ~iscell(imdbs)
    imdbs = {imdbs};
    roidbs = {roidbs};
end
imdbs = imdbs(:);
roidbs = roidbs(:);


image_roidb = {};

%%--- for computing bbox_means and bbox_stds ---
class_counts = zeros(1, 1) + eps;
sums = zeros(1, 4);
squared_sums = zeros(1, 4);

if test
    num_images = 10;
else
    num_images = length(imdbs{1}.image_ids);
end


save_path = [output_dir '/' 'rpn_prep_img_roidb_' type '/' mode];
if ~exist(save_path, 'dir')
    mkdir(save_path);
    fprintf('save_path folder created \n %s \n', save_path);
    % pause;
end

save_path_1 = [save_path '/' 'image_roidb_before_mean'];
if ~exist(save_path_1, 'dir')
    mkdir(save_path_1);
    fprintf('save_path_1 folder created \n %s \n', save_path_1);
    % pause;
end

save_path_2 = [save_path '/' 'image_roidb_final'];
if ~exist(save_path_2, 'dir')
    mkdir(save_path_2);
    fprintf('save_path_2 folder created \n %s \n', save_path_2);
    % pause;
end

if conf.target_only_gt
    for i=1:num_images
        image_roidb{1}(i,1).image_path = imdbs{1}.image_at(i);
        image_roidb{1}(i,1).image_id = imdbs{1}.image_ids{i};
        image_roidb{1}(i,1).im_size = imdbs{1}.sizes(i, :);
        image_roidb{1}(i,1).imdb_name = imdbs{1}.name;
        image_roidb{1}(i,1).num_classes = imdbs{1}.num_classes;
        image_roidb{1}(i,1).boxes = roidbs{1}.rois(i).boxes(roidbs{1}.rois(i).gt, :);
        image_roidb{1}(i,1).class = roidbs{1}.rois(i).class(roidbs{1}.rois(i).gt, :);
        image_roidb{1}(i,1).image = [];
        image_roidb{1}(i,1).bbox_targets = [];
        
        %--- append_bbox_regression_targets()---
        [anchors, im_scales] = proposal_locate_anchors(conf, image_roidb{1}(i,1).im_size);
        %function [anchors, im_scales] = proposal_locate_anchors(conf, im_size, target_scale, feature_map_size)
        im_scales = cell2mat(im_scales);
        anchors = cell2mat(anchors);
        
        bbox_targets = compute_targets(conf, ...
            scale_rois(image_roidb{1}(i,1).boxes, image_roidb{1}(i,1).im_size, im_scales), ...
            image_roidb{1}(i,1).class,  anchors, image_roidb{1}(i,1), im_scales);
        
        
        image_roidb{1}(i,1).bbox_targets = {bbox_targets};
        
        if ~(exist('bbox_means', 'var') && ~isempty(bbox_means) && exist('bbox_stds', 'var') && ~isempty(bbox_stds))
            for j = 1:length(conf.scales)
                targets =  image_roidb{1}(i,1).bbox_targets{j};
                gt_inds = find(targets(:, 1) > 0);
                if ~isempty(gt_inds)
                    class_counts = class_counts + length(gt_inds);
                    sums = sums + sum(targets(gt_inds, 2:end), 1);
                    squared_sums = squared_sums + sum(targets(gt_inds, 2:end).^2, 1);
                end
            end
        end
        
        image_roidb_st = image_roidb{1}(i,1);
        save_file = sprintf('%s/%06d.mat', save_path_1, i);
        save(save_file, 'image_roidb_st');
        
        if mod(i,1000) == 0
            fprintf('saving image_roidb for imageid = %06d at\n %s \n', i, save_file);
        end
        
    end
    
    if ~(exist('bbox_means', 'var') && ~isempty(bbox_means) && exist('bbox_stds', 'var') && ~isempty(bbox_stds))
        bbox_means = bsxfun(@rdivide, sums, class_counts);
        bbox_stds = (bsxfun(@minus, bsxfun(@rdivide, squared_sums, class_counts), bbox_means.^2)).^0.5;
        save([save_path '/' 'bbox_means.mat'], 'bbox_means');
        fprintf('bbox_means saved in \n %s \n', [save_path '/' 'bbox_means.mat']);
        save([save_path '/' 'bbox_stds.mat'], 'bbox_stds');
        fprintf('bbox_stds saved in \n %s \n', [save_path '/' 'bbox_stds.mat']);
    end
    
    
    
    
    % Normalize targets
    for i = 1:num_images
        for j = 1:length(conf.scales)
            targets =  image_roidb{1}(i,1).bbox_targets{j};
            gt_inds = find(targets(:, 1) > 0);
            if ~isempty(gt_inds)
                image_roidb{1}(i,1).bbox_targets{j}(gt_inds, 2:end) = ...
                    bsxfun(@minus, image_roidb{1}(i,1).bbox_targets{j}(gt_inds, 2:end), bbox_means);
                image_roidb{1}(i,1).bbox_targets{j}(gt_inds, 2:end) = ...
                    bsxfun(@rdivide, image_roidb{1}(i,1).bbox_targets{j}(gt_inds, 2:end), bbox_stds);
            end
        end
        
        image_roidb_st_2 = image_roidb{1}(i,1);
        save_file_2 = sprintf('%s/%06d.mat', save_path_2, i);
        save(save_file_2, 'image_roidb_st_2');
        if mod(i,1000) == 0
            fprintf('saving image_roidb for imageid = %06d at\n %s \n', i, save_file_2);
        end
    end
    
    
end



end


function bbox_targets = compute_targets(conf, gt_rois, gt_labels, ex_rois, image_roidb, im_scale)

% output: bbox_targets
%   positive: [class_label, regression_label]
%   ingore: [0, zero(regression_label)]
%   negative: [-1, zero(regression_label)]

if isempty(gt_rois)
    bbox_targets = zeros(size(ex_rois, 1), 5, 'double');
    bbox_targets(:, 1) = -1;
    return;
end

% ensure gt_labels is in single
gt_labels = single(gt_labels);
assert(all(gt_labels > 0));

% calc overlap between ex_rois(anchors) and gt_rois
ex_gt_overlaps = boxoverlap(ex_rois, gt_rois);

% drop anchors which run out off image boundaries, if necessary
if conf.drop_boxes_runoff_image
    contained_in_image = is_contain_in_image(ex_rois, round(image_roidb.im_size * im_scale));
    ex_gt_overlaps(~contained_in_image, :) = 0;
end

% for each ex_rois(anchors), get its max overlap with all gt_rois
[ex_max_overlaps, ex_assignment] = max(ex_gt_overlaps, [], 2);

% for each gt_rois, get its max overlap with all ex_rois(anchors), the
% ex_rois(anchors) are recorded in gt_assignment
% gt_assignment will be assigned as positive
% (assign a rois for each gt at least)
[gt_max_overlaps, gt_assignment] = max(ex_gt_overlaps, [], 1);

% ex_rois(anchors) with gt_max_overlaps maybe more than one, find them
% as (gt_best_matches)
[gt_best_matches, gt_ind] = find(bsxfun(@eq, ex_gt_overlaps, [gt_max_overlaps]));

% Indices of examples for which we try to make predictions
% both (ex_max_overlaps >= conf.fg_thresh) and gt_best_matches are
% assigned as positive examples
fg_inds = unique([find(ex_max_overlaps >= conf.fg_thresh); gt_best_matches]);

% Indices of examples for which we try to used as negtive samples
% the logic for assigning labels to anchors can be satisfied by both the positive label and the negative label
% When this happens, the code gives the positive label precedence to
% pursue high recall
bg_inds = setdiff(find(ex_max_overlaps < conf.bg_thresh_hi & ex_max_overlaps >= conf.bg_thresh_lo), ...
    fg_inds);

if conf.drop_boxes_runoff_image
    contained_in_image_ind = find(contained_in_image);
    fg_inds = intersect(fg_inds, contained_in_image_ind);
    bg_inds = intersect(bg_inds, contained_in_image_ind);
end

% Find which gt ROI each ex ROI has max overlap with:
% this will be the ex ROI's gt target
target_rois = gt_rois(ex_assignment(fg_inds), :);
src_rois = ex_rois(fg_inds, :);

% we predict regression_label which is generated by an un-linear
% transformation from src_rois and target_rois
[regression_label] = fast_rcnn_bbox_transform(src_rois, target_rois);

bbox_targets = zeros(size(ex_rois, 1), 5, 'double');
bbox_targets(fg_inds, :) = [gt_labels(ex_assignment(fg_inds)), regression_label];
bbox_targets(bg_inds, 1) = -1;

if 0 % debug
    %%%%%%%%%%%%%%
    im = imread(image_roidb.image_path);
    [im, im_scale] = prep_im_for_blob(im, conf.image_means, conf.scales, conf.max_size);
    imshow(mat2gray(im));
    hold on;
    cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'r'), ...
        num2cell(src_rois, 2));
    cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'g'), ...
        num2cell(target_rois, 2));
    hold off;
    %%%%%%%%%%%%%%
end

bbox_targets = sparse(bbox_targets);
end




function contained = is_contain_in_image(boxes, im_size)
contained = boxes >= 1 & bsxfun(@le, boxes, [im_size(2), im_size(1), im_size(2), im_size(1)]);

contained = all(contained, 2);
end

function scaled_rois = scale_rois(rois, im_size, im_scale)

% save('/mnt/sun-alpha/test_2_2.mat', 'im_scale'); % comment

im_size_scaled = round(im_size * im_scale);
scale = (im_size_scaled - 1) ./ (im_size - 1);
scaled_rois = bsxfun(@times, rois-1, [scale(2), scale(1), scale(2), scale(1)]) + 1;
end

% if 0
%     % this is for ref
%     image_roidb = ...
%         cellfun(@(x, y) ... // @(imdbs, roidbs)
%         arrayfun(@(z) ... //@([1:length(x.image_ids)])
%         struct( 'image_path', x.image_at(z), ...
%         'image_id', x.image_ids{z}, ...
%         'im_size', x.sizes(z, :), ...
%         'imdb_name', x.name, ...
%         'num_classes', x.num_classes, ...
%         'boxes', y.rois(z).boxes(y.rois(z).gt, :),...
%         'class', y.rois(z).class(y.rois(z).gt, :), ...
%         'image', [], ...
%         'bbox_targets', []), ...
%         [1:length(x.image_ids)]', 'UniformOutput', true),...
%         imdbs, roidbs, 'UniformOutput', false);
% end