% -------------------------------------------------------------------
% Author: Suman Saha
% This script is adapted from Faster R-CNN(*) matlab code repository:
% https://github.com/ShaoqingRen/faster_rcnn
% -------------------------------------------------------------------
% fast_rcnn_prepare_image_roidb_manual()
% for each iteration prepare image roidb from the rpn proposals saved
% at location: [output_dir '/rpn_prop_for_frcnn_training_' type '/filtered_aboxes'];
% it also cache image_roidb to disk so that if the function is called in
% future for a particular video frame which have already image_roidb
% computed, it does not again compute the image_roidb, rather it reads from
% the disk and return the image_roidb, usually for each fast rcnn iteration
% 2 images are processed referred by: sub_db_inds(1) and sub_db_inds(2)
% -------------------------------------------------------------------
function [image_roidb, bbox_means, bbox_stds] = fast_rcnn_prepare_image_roidb_manual(conf, imdbs, rois, output_dir, sub_db_inds, type, bbox_means, bbox_stds)
if ~exist('bbox_means', 'var')
    bbox_means = [];
    bbox_stds = [];
end
if ~iscell(imdbs)
    imdbs = {imdbs};
end
imdbs = imdbs(:);
% cache_dir = fullfile(output_dir, 'output', 'rpn_cachedir', cache_name, imdbs{1}.name);
% image_roidb_cache_path_ = [cache_dir '/' 'input_region_props_for_frcnn' '/' 'image_roidb'];

image_roidb_cache_path = [output_dir '/' 'frcnn_prep_img_roidb_' type];
if ~exist(image_roidb_cache_path, 'dir')
    mkdir(image_roidb_cache_path);
end

assert(length(sub_db_inds) == conf.ims_per_batch);
done = false;
image_roidb_cache_file_1 = sprintf('%s/%06d.mat', image_roidb_cache_path, sub_db_inds(1));
image_roidb_cache_file_2 = sprintf('%s/%06d.mat', image_roidb_cache_path, sub_db_inds(2));
if exist(image_roidb_cache_file_1,'file') &&  exist(image_roidb_cache_file_2,'file')
    %image_roidb = struct();
    irst_1 = load(image_roidb_cache_file_1);
    image_roidb(1,1) = irst_1.img_roidb;
    irst_2 = load(image_roidb_cache_file_2);
    image_roidb(2,1) = irst_2.img_roidb;
    clear irst_1;
    clear irst_2;
    done = true;
end


ab_fetch_path = [output_dir '/rpn_prop_for_frcnn_training_' type '/filtered_aboxes'];

if ~done
    image_roidb = cell(length(sub_db_inds),1);
    % fprintf('generating image_roidb on the fly for img id = %06d and %06d\n', sub_db_inds(1), sub_db_inds(2));
    irc = 0;
    for i=sub_db_inds' %1:length(imdbs{1}.image_ids)
        irc=irc+1;
        
        ts = tic;
        ab_fetch_file = [ab_fetch_path '/' imdbs{1}.image_ids{i} '.mat'];
        if ~exist(ab_fetch_file,'file')
            fprintf('In roidb_from_proposal_manual.m --> error:file does not exist : \n %s \n', ab_fetch_file);
            fprintf('generate this file and run script again, pausing for now...\n');
            pause;
        end
        
        abst = load(ab_fetch_file);
        boxes = abst.aboxes(:,1:4);
        clear abst;
        
        is_gt = rois(i).gt;
        gt_boxes = rois(i).boxes(is_gt, :);
        gt_classes = rois(i).class(is_gt, :);
        all_boxes = cat(1, rois(i).boxes, boxes);
        
        num_gt_boxes = size(gt_boxes, 1);
        num_boxes = size(boxes, 1);
        
        overlap = cat(1, rois(i).overlap, zeros(num_boxes, size(rois(i).overlap, 2)));
        %boxes_ = cat(1, rois(i).boxes, boxes);
        class = cat(1, rois(i).class, zeros(num_boxes, 1));
        for j = 1:num_gt_boxes
            overlap(:, gt_classes(j)) = ...
                max(full(overlap(:, gt_classes(j))), boxoverlap(all_boxes, gt_boxes(j, :))); % function boxoverlap() is under utils/
        end
        
        image_roidb{1}(irc,1).image_path = imdbs{1}.image_at(i);
        image_roidb{1}(irc,1).image_id = imdbs{1}.image_ids{i};
        image_roidb{1}(irc,1).im_size = imdbs{1}.sizes(i, :);
        image_roidb{1}(irc,1).imdb_name = imdbs{1}.name;
        image_roidb{1}(irc,1).overlap = overlap;
        image_roidb{1}(irc,1).boxes = all_boxes;
        image_roidb{1}(irc,1).class = class;
        image_roidb{1}(irc,1).image = [];
        image_roidb{1}(irc,1).bbox_targets = [];
        
        te = toc(ts);
        % fprintf('\n generating image_roidb for imgid:= %06d  in %.3fs\n', i, te);
        
    end
    
    image_roidb = cat(1, image_roidb{:});
    
    % fprintf(' running append_bbox_regression_targets()... \n');
    % enhance roidb to contain bounding-box regression targets
    [image_roidb] = append_bbox_regression_targets(conf, image_roidb, bbox_means, bbox_stds);
    
    
    % image_roidb_cache
    irc = 0;
    for i=sub_db_inds'
        irc=irc+1;
        img_roidb = image_roidb(irc,1);  % img_roidb is a 1x1 structure
        image_roidb_cache_file = sprintf('%s/%06d.mat', image_roidb_cache_path, i );
        save(image_roidb_cache_file, 'img_roidb');
    end
    
end

end


%%---------------------------------------------------------------------
function [image_roidb] = append_bbox_regression_targets(conf, image_roidb, means, stds)
%%---------------------------------------------------------------------
% means and stds -- (k+1) * 4, include background class

num_images = length(image_roidb);
% Infer number of classes from the number of columns in gt_overlaps
num_classes = size(image_roidb(1).overlap, 2);
valid_imgs = true(num_images, 1);
for i = 1:num_images
    rois = image_roidb(i).boxes;
    [image_roidb(i).bbox_targets, valid_imgs(i)] = ...
        compute_targets(conf, rois, image_roidb(i).overlap);
end
if ~all(valid_imgs)
    image_roidb = image_roidb(valid_imgs);
    num_images = length(image_roidb);
    fprintf('Warning: fast_rcnn_prepare_image_roidb: filter out %d images, which contains zero valid samples\n', sum(~valid_imgs));
end

% Normalize targets
for i = 1:num_images
    targets = image_roidb(i).bbox_targets;
    for cls = 1:num_classes
        cls_inds = find(targets(:, 1) == cls);
        if ~isempty(cls_inds)
            image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                bsxfun(@minus, image_roidb(i).bbox_targets(cls_inds, 2:end), means(cls+1, :));
            image_roidb(i).bbox_targets(cls_inds, 2:end) = ...
                bsxfun(@rdivide, image_roidb(i).bbox_targets(cls_inds, 2:end), stds(cls+1, :));
        end
    end
end
end


%%---------------------------------------------------------------------
function [bbox_targets, is_valid] = compute_targets(conf, rois, overlap)
%%---------------------------------------------------------------------
overlap = full(overlap);

[max_overlaps, max_labels] = max(overlap, [], 2);

% ensure ROIs are floats
rois = single(rois);

bbox_targets = zeros(size(rois, 1), 5, 'single');

% Indices of ground-truth ROIs
gt_inds = find(max_overlaps == 1);

if ~isempty(gt_inds)
    % Indices of examples for which we try to make predictions
    ex_inds = find(max_overlaps >= conf.bbox_thresh);
    
    % Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = boxoverlap(rois(ex_inds, :), rois(gt_inds, :));
    
    assert(all(abs(max(ex_gt_overlaps, [], 2) - max_overlaps(ex_inds)) < 10^-4));
    
    % Find which gt ROI each ex ROI has max overlap with:
    % this will be the ex ROI's gt target
    [~, gt_assignment] = max(ex_gt_overlaps, [], 2);
    gt_rois = rois(gt_inds(gt_assignment), :);
    ex_rois = rois(ex_inds, :);
    
    [regression_label] = fast_rcnn_bbox_transform(ex_rois, gt_rois);
    
    bbox_targets(ex_inds, :) = [max_labels(ex_inds), regression_label];
end

% Select foreground ROIs as those with >= fg_thresh overlap
is_fg = max_overlaps >= conf.fg_thresh;
% Select background ROIs as those within [bg_thresh_lo, bg_thresh_hi)
is_bg = max_overlaps < conf.bg_thresh_hi & max_overlaps >= conf.bg_thresh_lo;

% check if there is any fg or bg sample. If no, filter out this image
is_valid = true;
if ~any(is_fg | is_bg)
    is_valid = false;
end
end

