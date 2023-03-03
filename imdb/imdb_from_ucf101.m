% -------------------------------------------------------------------
% Author: Suman Saha
% This script is adapted from Faster R-CNN(*) matlab code repository:
% https://github.com/ShaoqingRen/faster_rcnn
% -------------------------------------------------------------------

function imdb = imdb_from_ucf101(image_set, flip, imdb_roidb_cache_dir, img_dir, num_classes)
cache_file = [imdb_roidb_cache_dir '/' 'imdb_ucf101_' image_set];
if flip
    cache_file = [cache_file, '_flip'];
end
try
    load(cache_file);
catch
    
    imdb.name = ['ucf101_' image_set];
    imdb.image_dir = img_dir;
    imdb.image_ids =  getUCF101ImageIds(image_set, imdb_roidb_cache_dir);
    imdb.extension = 'jpg';
    imdb.flip = flip;
    
    % this will execute for trainsplit01 when horizontal flipping is to be done in roidb
    if flip
        image_at = @(i) sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
        flip_image_at = @(i) sprintf('%s/%s_flip.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
        for i = 1:length(imdb.image_ids)
            if ~exist(flip_image_at(i), 'file')
                im = imread(image_at(i));
                imwrite(fliplr(im), flip_image_at(i));
                fprintf('flipping and saving flipped images to: \n %s \n', flip_image_at(i));
            else
                fprintf('already flipped \n %s \n', flip_image_at(i));
            end
        end
        img_num = length(imdb.image_ids)*2;
        image_ids = imdb.image_ids;
        imdb.image_ids(1:2:img_num) = image_ids;
        imdb.image_ids(2:2:img_num) = cellfun(@(x) [x, '_flip'], image_ids, 'UniformOutput', false);
        imdb.flip_from = zeros(img_num, 1);
        imdb.flip_from(2:2:img_num) = 1:2:img_num;
    end
    
    imdb.classes =  getUCF101Classes(num_classes); % VOCopts.classes;
    imdb.num_classes = length(imdb.classes);
    imdb.class_to_id = ...
        containers.Map(imdb.classes, 1:imdb.num_classes);
    imdb.class_ids = 1:imdb.num_classes;
    % VOC specific functions for evaluation and region of interest DB
    imdb.eval_func = @imdb_eval_ucf101; %@imdb_eval_voc;
    imdb.roidb_func = @roidb_from_ucf101; % @roidb_from_voc;
    imdb.image_at = @(i) ...
        sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
    for i = 1:length(imdb.image_ids)
        tic_toc_print('imdb (%s): %d/%d\n', imdb.name, i, length(imdb.image_ids));
        
        % info = imfinfo(sprintf(VOCopts.imgpath, imdb.image_ids{i}));
        info = imfinfo(sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension));
        imdb.sizes(i, :) = [info.Height info.Width];
    end
    fprintf('Saving imdb to cache...');
    % save(cache_file, 'imdb', '-v7.3');
    save(cache_file, 'imdb');
    fprintf('done\n');
end
