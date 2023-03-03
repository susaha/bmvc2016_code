% -------------------------------------------------------------------
% Author: Suman Saha
% This script is adapted from Faster R-CNN(*) matlab code repository: 
% https://github.com/ShaoqingRen/faster_rcnn
% -------------------------------------------------------------------
function dataset = ucf101_trainval(dataset, usage, use_flip, imdb_roidb_cache_dir, img_dir, num_classes)
switch usage
    case {'train'}
        imageset = 'trainsplit01';
        dataset.imdb_train    = {  imdb_from_ucf101(imageset, use_flip, imdb_roidb_cache_dir, img_dir, num_classes) };
        imdb_roidb_cache_dir = {imdb_roidb_cache_dir};
        dataset.roidb_train   = cellfun(@(x,y) x.roidb_func(x,y), dataset.imdb_train, imdb_roidb_cache_dir, 'UniformOutput', false);
        
    case {'validation'}
        imageset = 'validation-split01';
        dataset.imdb_test     = imdb_from_ucf101(imageset,  use_flip, imdb_roidb_cache_dir, img_dir, num_classes) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test, imdb_roidb_cache_dir);
    otherwise
        error('usage = ''train'' or ''validation''');
end

end