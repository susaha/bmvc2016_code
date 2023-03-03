% -------------------------------------------------------------------
% Author: Gurkirt Singh
% Main evalutation script to compute mAP
% -------------------------------------------------------------------
function [mAP,mAIoU, AP] = get_PR_curve(video, xmldata, actions, iou_th)
% load(xmlfile)
num_vid = length(xmldata);
num_actions = length(actions);
AP = zeros(num_actions,1);
averageIoU = zeros(num_actions,1);

cc = zeros(num_actions,1);
for a=1:num_actions
    allscore{a} = zeros(100000,2,'single');
end

total_num_gt_tubes = zeros(num_actions,1); % count all the gt tubes from all the vidoes for label a

for vid=1:num_vid
    
    %--- use these two line if your xmldata(vid).videoName contains only the video name e.g. v_Basketball_g01_c01
    [action, ~] = getActionNVidNameV2(xmldata(vid).videoName); % here you pass the video name and it extracts the action name e.g. v_Basketball_g01_c01
    gtVidInd = getGtVidIndV2(video,[action,'/',xmldata(vid).videoName]); %  

    %--- use this single line if your xmldata(vid).videoName contains action/videoname the video name e.g. Basketball/v_Basketball_g01_c01
    %--- this is much more expensive than the previous two lines as it performs multiple strsplit() operations for each vid-th iteration times the length(video)
    %--- within getGtVidIndV1(video,xmldata(vid).videoName); function, so try to avoid this as it is really slows down your evaluation
    % gtVidInd = getGtVidIndV1(video,xmldata(vid).videoName);
    
    %--- this for sanity check---
    if gtVidInd==-1
        disp(xmldata(vid).videoName);
        disp('error');
    end
    fprintf('accumulating detection scores and tp and fp labels (1 or 0) for video: %s\n', xmldata(vid).videoName);
    
    % sorting the detections in decreasing order 
    dt_tubes = sort_detection(xmldata(vid));       
    gt_tubes = video(gtVidInd).tubes;
    
    
    num_detection = length(dt_tubes.class);
    num_gt_tubes = length(gt_tubes);
    
    for gtind = 1:num_gt_tubes
        action_id = gt_tubes(gtind).class;
        total_num_gt_tubes(action_id) = total_num_gt_tubes(action_id) + 1;
    end
    dt_labels = dt_tubes.class;
    covered_gt_tubes = zeros(num_gt_tubes,1);
    
    for dtind = 1:num_detection
        dt_fnr = dt_tubes.framenr(dtind).fnr;
        dt_bb = dt_tubes.boxes(dtind).bxs;
        dt_label = dt_labels(dtind);
        
        cc(dt_label) = cc(dt_label) + 1;
        
        ioumax=-inf;maxgtind=0;
        for gtind = 1:num_gt_tubes
            action_id = gt_tubes(gtind).class;
            if ~covered_gt_tubes(gtind) && dt_label == action_id
                gt_fnr = gt_tubes(gtind).sf:gt_tubes(gtind).ef;
                gt_bb = gt_tubes(gtind).boxes;
                iou = compute_spatio_temporal_iou(gt_fnr, gt_bb, dt_fnr, dt_bb);
                if iou>ioumax
                    ioumax=iou;
                    maxgtind=gtind;
                end
            end
        end
        if ioumax>iou_th
            covered_gt_tubes(maxgtind) = 1;
            allscore{dt_label}(cc(dt_label),:) = [dt_tubes.score(dtind),1];
            averageIoU(dt_label) = averageIoU(dt_label) + ioumax;
        else
            allscore{dt_label}(cc(dt_label),:) = [dt_tubes.score(dtind),0];
        end
    end
end

for a=1:num_actions
    allscore{a} = allscore{a}(1:cc(a),:);
    scores = allscore{a}(:,1);
    labels = allscore{a}(:,2);
    [~, si] = sort(scores,'descend');
    %     scores = scores(si);
    labels = labels(si);
    fp=cumsum(labels==0);
    tp=cumsum(labels==1);
    cdet =0;
    if ~isempty(tp)>0
        cdet = tp(end);
        averageIoU(a) = (averageIoU(a)+0.000001)/(tp(end)+0.00001);
    end
    
    recall=tp/total_num_gt_tubes(a);
    precision=tp./(fp+tp);
    AP(a) = xVOCap(recall,precision);
    
    fprintf('Action %02d AP = %0.5f and AIOU %0.5f GT %03d total det %02d correct det %02d %s\n', a, AP(a),averageIoU(a),total_num_gt_tubes(a),length(tp),cdet,actions{a});
    
    draw = 0;
    if draw
        % plot precision/recall
        plot(recall,precision,'-');
        grid;
        xlabel 'recall'
        ylabel 'precision'
        title(sprintf('class: %s, AP = %.3f',actions{a},AP(a)));
    end
end


mAP  = mean(AP);
averageIoU(isnan(averageIoU)) = 0;
mAIoU = mean(averageIoU);
end

%------------------------------------------------------------------------------------------------------------------------------------------------
function [action, videoName] = getActionNVidNameV2(str)
%------------------------------------------------------------------------------------------------------------------------------------------------
indx = strsplit(str, '_');
action = indx{2};
videoName = str;
end

%------------------------------------------------------------------------------------------------------------------------------------------------
function [action, videoName] = getActionNVidNameV1(str)
%------------------------------------------------------------------------------------------------------------------------------------------------
indx = strfind(str, '/');
action = str(1:indx-1);
videoName = str(indx+1:end);
end


function gtVidInd = getGtVidIndV1(video,videoName)
gtVidInd = -1;
for i=1:length(video)
    vidid = video(i).name;
    vidid = strsplit(vidid,'/');
    vidid = vidid{2};
    if strcmp(vidid,videoName)
        gtVidInd = i;
        break;
    end
end
end

function gtVidInd = getGtVidIndV2(video,videoName)
gtVidInd = -1;
for i=1:length(video)
    vidid = video(i).name;
%     vidid = strsplit(vidid,'/');
%     vidid = vidid{2};
    if strcmp(vidid,videoName)
        gtVidInd = i;
        break;
    end
end
end

function sorted_tubes = sort_detection(dt_tubes)
sorted_tubes = dt_tubes;
if length(dt_tubes.class)>0    
    num_detection = length(dt_tubes.class);
    scores = dt_tubes.score;
    [~,indexs] = sort(scores,'descend');
    for dt = 1 : num_detection
        dtind = indexs(dt);
        sorted_tubes.framenr(dt).fnr = dt_tubes.framenr(dtind).fnr;
        sorted_tubes.boxes(dt).bxs = dt_tubes.boxes(dtind).bxs;
        sorted_tubes.class(dt) = dt_tubes.class(dtind);
        sorted_tubes.score(dt) = dt_tubes.score(dtind);
        sorted_tubes.nr(dt) = dt;
    end
end
end

