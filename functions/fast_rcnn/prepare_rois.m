function rois = prepare_rois(imdbs, roidbs)
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
end