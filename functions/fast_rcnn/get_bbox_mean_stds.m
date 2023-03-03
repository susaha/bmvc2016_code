% -------------------------------------------------------------------
% Author: Suman Saha
% This script is adapted from Faster R-CNN(*) matlab code repository: 
% https://github.com/ShaoqingRen/faster_rcnn
% -------------------------------------------------------------------

function get_bbox_mean_stds(conf, imdbs, roidbs, cache_name, output_dir, test, portion_of_entire_trainset, bbox_means, bbox_stds)

TEST_FALG = test;
% portion_of_entire_trainset = 1; %0.3; % 30 %

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



assert(strcmp(roidbs{1}.name, imdbs{1}.name));
rois = roidbs{1}.rois;

keep_raw_proposal = false;
if ~keep_raw_proposal
    % remove proposal boxes in roidb
    for i = 1:length(rois)
        is_gt = rois(i).gt;
        rois(i).gt = rois(i).gt(is_gt, :);
        rois(i).overlap = rois(i).overlap(is_gt, :);
        rois(i).boxes = rois(i).boxes(is_gt, :);
        rois(i).class = rois(i).class(is_gt, :);
    end
end

image_roidb_bbox_mean_save_path = [output_dir '/' 'rpn_prop_for_frcnn_training_spatial' '/' 'image_roidb'];
if ~exist(image_roidb_bbox_mean_save_path, 'dir')
    mkdir(image_roidb_bbox_mean_save_path);
    fprintf('created folder\n %s \n here mean and stds will be saved\n press any key to continue...\n', image_roidb_bbox_mean_save_path);    
end

% 
ab_fetch_path = [output_dir '/' 'rpn_prop_for_frcnn_training_spatial' '/' 'filtered_aboxes'];
fprintf('ab_fetch_path= \n %s \n from here the filter aboxes will be read \n', ab_fetch_path);


list = dir([ab_fetch_path '/*.mat']);
%assert(length(list) == length(rois) );


image_roidb = {};

t_500 = 0;

if TEST_FALG
    num_imgs = 100;
    max_iter = 100;
else
    num_imgs = length(imdbs{1}.image_ids);
    num_img_sub = round(num_imgs*portion_of_entire_trainset);
    rind = randperm(num_imgs, num_img_sub);
    max_iter = length(rind);
end


fprintf('%d images will be sampled to compute bbox mean and stds\n', max_iter);
% rand_ind = randperm(length(dataset.videos));
% i = rand_ind(ii);

%for i=1:length(imdbs{1}.image_ids)
for k=1:max_iter
    
    if TEST_FALG
        i = k;
    else
        i = rind(k);
    end
    ts = tic;
    [~, image_name1] = fileparts(imdbs{1}.image_ids{i});
    str = list(i).name;
    ind = strfind(str,'.');
    image_name2 = str(1:ind-1);
    assert(strcmp(image_name1, image_name2));
    
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
    
    image_roidb{1}(k,1).image_path = imdbs{1}.image_at(i);
    image_roidb{1}(k,1).image_id = imdbs{1}.image_ids{i};
    image_roidb{1}(k,1).im_size = imdbs{1}.sizes(i, :);
    image_roidb{1}(k,1).imdb_name = imdbs{1}.name;
    image_roidb{1}(k,1).overlap = overlap;
    image_roidb{1}(k,1).boxes = all_boxes;
    image_roidb{1}(k,1).class = class;
    image_roidb{1}(k,1).image = [];
    image_roidb{1}(k,1).bbox_targets = [];
    
    t_500 = t_500 + toc(ts);
    
    if mod(k,1000) == 0
        fprintf('\ntime taken for 1000 image frames = %.3fs\n img_count=%06d img_id=%06d total _frames=%d \n', t_500, k, i, max_iter);
        t_500 = 0;
    elseif k==1
        fprintf('\ncurrent frame = %06d\n', i);
    end
end

image_roidb = cat(1, image_roidb{:});

fprintf(' running append_bbox_regression_targets()... \n');
% enhance roidb to contain bounding-box regression targets
[image_roidb, bbox_means, bbox_stds] = append_bbox_regression_targets(conf, image_roidb, bbox_means, bbox_stds);


save_file_bbox_means = [image_roidb_bbox_mean_save_path '/' 'bbox_means' '.mat'];
save_file_bbox_stds = [image_roidb_bbox_mean_save_path '/' 'bbox_stds' '.mat'];



try
    save(save_file_bbox_means, 'bbox_means');
    fprintf('bbox_means saved at \n %s \n', save_file_bbox_means);
catch
    try
        save(save_file_bbox_means, 'bbox_means', '-v7.3');
        fprintf('bbox_means saved at \n %s \n', save_file_bbox_means);
    catch
        fprintf('error savining bbox_means');
        pause;
    end
end

try
    save(save_file_bbox_stds, 'bbox_stds');
    fprintf('bbox_stds saved at \n %s \n', save_file_bbox_stds);
catch
    try
        save(save_file_bbox_stds, 'bbox_stds', '-v7.3');
        fprintf('bbox_stds saved at \n %s \n', save_file_bbox_stds);
    catch
        fprintf('error savining bbox_stds');
        pause;
    end
end


end


%%---------------------------------------------------------------------
function [image_roidb, means, stds] = append_bbox_regression_targets(conf, image_roidb, means, stds)
%%---------------------------------------------------------------------
% means and stds -- (k+1) * 4, include background class

num_images = length(image_roidb);
% Infer number of classes from the number of columns in gt_overlaps
num_classes = size(image_roidb(1).overlap, 2);
valid_imgs = true(num_images, 1);
for i = 1:num_images
    rois = image_roidb(i).boxes;
    
    ts = tic;
    [image_roidb(i).bbox_targets, valid_imgs(i)] = ...
        compute_targets(conf, rois, image_roidb(i).overlap);    
    te = toc(ts);
    fprintf('compute_targets() : img count = %d  time taken = %.3fs\n', i, te);
end
if ~all(valid_imgs)
    image_roidb = image_roidb(valid_imgs);
    num_images = length(image_roidb);
    fprintf('Warning: fast_rcnn_prepare_image_roidb: filter out %d images, which contains zero valid samples\n', sum(~valid_imgs));
end

if ~(exist('means', 'var') && ~isempty(means) && exist('stds', 'var') && ~isempty(stds))
    % Compute values needed for means and stds
    % var(x) = E(x^2) - E(x)^2
    class_counts = zeros(num_classes, 1) + eps;
    sums = zeros(num_classes, 4);
    squared_sums = zeros(num_classes, 4);
    for i = 1:num_images
        ts1 = tic;
        targets = image_roidb(i).bbox_targets;
        for cls = 1:num_classes
            cls_inds = find(targets(:, 1) == cls);
            if ~isempty(cls_inds)
                class_counts(cls) = class_counts(cls) + length(cls_inds);
                sums(cls, :) = sums(cls, :) + sum(targets(cls_inds, 2:end), 1);
                squared_sums(cls, :) = squared_sums(cls, :) + sum(targets(cls_inds, 2:end).^2, 1);
            end
        end
        te1 = toc(ts1);
        fprintf('compute variance : img count = %d  time taken = %.3fs\n', i, te1);
    end
    
    means = bsxfun(@rdivide, sums, class_counts);
    stds = (bsxfun(@minus, bsxfun(@rdivide, squared_sums, class_counts), means.^2)).^0.5;
    
    % add background class
    means = [0, 0, 0, 0; means];
    stds = [0, 0, 0, 0; stds];
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






