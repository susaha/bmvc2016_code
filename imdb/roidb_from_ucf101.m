% -------------------------------------------------------------------
% Author: Suman Saha
% This script is adapted from Faster R-CNN(*) matlab code repository:
% https://github.com/ShaoqingRen/faster_rcnn
% -------------------------------------------------------------------
function roidb = roidb_from_ucf101(imdb, imdb_roidb_cache_dir, varargin)
ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addRequired('imdb_roidb_cache_dir', @isstr);
ip.addParamValue('exclude_difficult_samples',       true,   @islogical);
ip.addParamValue('with_selective_search',           false,  @islogical);
ip.addParamValue('with_edge_box',                   false,  @islogical);
ip.addParamValue('with_self_proposal',              false,  @islogical);
ip.addParamValue('rootDir',                         '.',    @ischar);
ip.addParamValue('extension',                       '',     @ischar);
ip.parse(imdb, imdb_roidb_cache_dir, varargin{:});
opts = ip.Results;
roidb.name = imdb.name;
%###### START: The below part is not executed as per the present opts values #####
if ~isempty(opts.extension)
    opts.extension = ['_', opts.extension];
end
regions_file_ss = fullfile(opts.rootDir, sprintf('/data/selective_search_data/%s%s.mat', roidb.name, opts.extension));
regions_file_eb = fullfile(opts.rootDir, sprintf('/data/edge_box_data/%s%s.mat', roidb.name, opts.extension));
regions_file_sp = fullfile(opts.rootDir, sprintf('/data/self_proposal_data/%s%s.mat', roidb.name, opts.extension));

cache_file_ss = [];
cache_file_eb = [];
cache_file_sp = [];
if opts.with_selective_search
    cache_file_ss = 'ss_';
    if~exist(regions_file_ss, 'file')
        error('roidb_from_ilsvrc:: cannot find %s', regions_file_ss);
    end
end

if opts.with_edge_box
    cache_file_eb = 'eb_';
    if ~exist(regions_file_eb, 'file')
        error('roidb_from_ilsvrc:: cannot find %s', regions_file_eb);
    end
end

if opts.with_self_proposal
    cache_file_sp = 'sp_';
    if ~exist(regions_file_sp, 'file')
        error('roidb_from_ilsvrc:: cannot find %s', regions_file_sp);
    end
end
%###### END: The above part is not executed as per the present opts values #####

cache_file = fullfile([imdb_roidb_cache_dir '/roidb_' cache_file_ss cache_file_eb cache_file_sp imdb.name opts.extension]);

if imdb.flip
    cache_file = [cache_file '_flip'];
end
if opts.exclude_difficult_samples
    cache_file = [cache_file '_easy'];
end
cache_file = [cache_file, '.mat'];
try
    load(cache_file);
catch
    
    if strcmp('ucf101_trainsplit01', imdb.name)
        load([imdb_roidb_cache_dir '/' 'ucf101_annotation_trainsplit01.mat']);  % annotation
    elseif strcmp('ucf101_validation-split01', imdb.name)
        load([imdb_roidb_cache_dir '/' 'ucf101_annotation_validation-split01.mat']); % annotation
    end
    
    roidb.name = imdb.name;
    fprintf('Loading region proposals...');
    regions = [];
    
    
    %###### START: The below part is not executed as per the present opts values #####
    if opts.with_selective_search
        regions = load_proposals(regions_file_ss, regions);
    end
    if opts.with_edge_box
        regions = load_proposals(regions_file_eb, regions);
    end
    if opts.with_self_proposal
        regions = load_proposals(regions_file_sp, regions);
    end
    fprintf('done\n');
    %###### END: The below part is not executed as per the present opts values #####
        
    
    if isempty(regions)
        fprintf('Warrning: no windows proposal is loaded !\n');
        regions.boxes = cell(length(imdb.image_ids), 1);
        if imdb.flip
            regions.images = imdb.image_ids(1:2:end);
        else
            regions.images = imdb.image_ids;
        end
    end
    
    num_class = length(imdb.classes);
    img_width = imdb.sizes(1, 2);
    
    if ~imdb.flip % this is for testsplit01
        for i = 1:length(imdb.image_ids)
            tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids));
            if ~isempty(regions)
                [~, image_name1] = fileparts(imdb.image_ids{i});
                [~, image_name2] = fileparts(regions.images{i});
                assert(strcmp(image_name1, image_name2));
            end
            
            annot_img_id = annotation{i,2};
            assert(strcmp(image_name1, annot_img_id));
            gt_boxes = annotation{i,1};
            % conv from [x y w h] to [x1 y1 x2 y2] required by attach_proposals()
            gt_boxes =  [gt_boxes(:,1:2) gt_boxes(:,1:2)+gt_boxes(:,3:4)];
            gt_class = annotation{i,3};
            
            roidb.rois(i) = attach_proposals(gt_boxes, gt_class, num_class, img_width, regions.boxes{i},  false);
        end
        
    else  % this is for trainsplit01
        for i = 1:length(imdb.image_ids)/2
            tic_toc_print('roidb (%s): %d/%d\n', roidb.name, i, length(imdb.image_ids)/2);
            
            if ~isempty(regions)
                [~, image_name1] = fileparts(imdb.image_ids{i*2-1});
                [~, image_name2] = fileparts(regions.images{i});
                assert(strcmp(image_name1, image_name2));
                assert(imdb.flip_from(i*2) == i*2-1);
            end
            
            annot_img_id =  annotation{i,2};
            assert(strcmp(image_name1, annot_img_id));
            gt_boxes = annotation{i,1};
            % conv from [x y w h] to [x1 y1 x2 y2] required by attach_proposals()
            gt_boxes =  [gt_boxes(:,1:2) gt_boxes(:,1:2)+gt_boxes(:,3:4)];
            gt_class = annotation{i,3};
            
            
            
            roidb.rois(i*2-1) = attach_proposals(gt_boxes, gt_class, num_class, img_width, regions.boxes{i},  false);
            roidb.rois(i*2) = attach_proposals(gt_boxes, gt_class, num_class, img_width, regions.boxes{i},  true);
        end
    end
    
    fprintf('Saving roidb to cache...');
    save(cache_file, 'roidb');
    % save(cache_file, 'roidb', '-v7.3');
    fprintf('done\n');
end


% ------------------------------------------------------------------------
% -- if you use Fast rcnn - that is SS boxes or Edgeboxes to train the
% Fast-RCNN then boxes will there otherwise boxes will be emtpy
% I am keeping it , useful to test SS-boxes and EdgeBoxes
function rec = attach_proposals(gt_boxes, gt_class, num_class, img_width, boxes, flip)
% ------------------------------------------------------------------------

% change selective search order from [y1 x1 y2 x2] to [x1 y1 x2 y2]

%###### START: The below part is not executed as per the present opts values #####
if ~isempty(boxes)
    boxes = boxes(:, [2 1 4 3]);
    if flip
        boxes(:, [1, 3]) = voc_rec.imgsize(1) + 1 - boxes(:, [3, 1]);
    end
end
%###### END: The above part is not executed as per the present opts values #####



%####### here you need to assign values for gt_boxes, all_boxes etc.#######
% voc_rec.objects(valid_objects).bbox is in format [x1 y1 x2 y2] ,
% in place of voc_rec.objects(valid_objects).bbox , you need to pass ucf101
% gt_boxes for this frame and change it to x1 y1 x2 y2 format
% gt_boxes = cat(1, voc_rec.objects(valid_objects).bbox);


if flip
    % in place voc_rec.imgsize(1) you need to pu the width of ucf101 image
    % = 320 , beacuse this horizontal flipping
    gt_boxes(:, [1, 3]) = img_width + 1 - gt_boxes(:, [3, 1]);
end

all_boxes = cat(1, gt_boxes, boxes);

% here you need to create a [num_gt_boxes x 1] double with class id, e.g.
% [4 4]' etc
%gt_classes = class_to_id.values({voc_rec.objects(valid_objects).class});
% gt_classes = cat(1, gt_classes{:});


num_gt_boxes = size(gt_boxes, 1);

num_boxes = size(boxes, 1);

rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));

% in place of class_to_id.Count, put 24 or any variable num_cls = 24
rec.overlap = zeros(num_gt_boxes+num_boxes, num_class, 'single');


for i = 1:num_gt_boxes
    rec.overlap(:, gt_class) = max(rec.overlap(:, gt_class), boxoverlap(all_boxes, gt_boxes(i, :)));
end


rec.boxes = single(all_boxes);
rec.feat = [];

gt_class_ = [];
for i=1:length(rec.gt)
    gt_class_ = cat(1, gt_class_, gt_class);
end
rec.class = uint8(cat(1, gt_class_, zeros(num_boxes, 1)));


% ------------------------------------------------------------------------
function regions = load_proposals(proposal_file, regions)
% ------------------------------------------------------------------------
if isempty(regions)
    regions = load(proposal_file);
else
    regions_more = load(proposal_file);
    if ~all(cellfun(@(x, y) strcmp(x, y), regions.images(:), regions_more.images(:), 'UniformOutput', true))
        error('roidb_from_ilsvrc: %s is has different images list with other proposals.\n', proposal_file);
    end
    regions.boxes = cellfun(@(x, y) [double(x); double(y)], regions.boxes(:), regions_more.boxes(:), 'UniformOutput', false);
end
