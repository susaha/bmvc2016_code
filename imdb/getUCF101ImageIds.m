% -------------------------------------------------------------------
% Author: Suman Saha
% -------------------------------------------------------------------
function [image_ids] = getUCF101ImageIds(image_set, imdb_roidb_cache_dir)
if strcmp(image_set,'trainsplit01')
    train_imi = load([imdb_roidb_cache_dir '/ucf101_trainsplit01_imdb_inputs.mat']); % imdb_inputs
    img_id = train_imi.imdb_inputs.img_id;
    image_ids = cell(img_id,1);
    for i=1:img_id
        name = sprintf('%06d', i);
        image_ids{i,1} = name;
    end
elseif strcmp(image_set,'validation-split01')
    train_imi = load([imdb_roidb_cache_dir '/ucf101_trainsplit01_imdb_inputs.mat']); % imdb_inputs
    test_imi = load([imdb_roidb_cache_dir '/ucf101_validation-split01_imdb_inputs.mat']); % imdb_inputs
    sid = train_imi.imdb_inputs.img_id;
    eid = test_imi.imdb_inputs.img_id;
    dif =  eid - sid;
    image_ids = cell(dif,1);
    j=1;
    for i=sid+1:1:eid
        name = sprintf('%06d', i);
        image_ids{j,1} = name;
        j=j+1;
    end
else
    error('pass the image_set parameter value to this function correctly!');
end
