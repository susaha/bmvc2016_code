% -------------------------------------------------------------------
% Author: Gurkirt Singh
% -------------------------------------------------------------------
function xmld = tube2Xml(tubes, min_num_frames, topk, videos)

vids = cell(length(videos),1);

for i=1:length(videos)
    str = videos(i).name;
    str = strsplit(str,'/');
    vids{i} = str{2};
end
%vids = unique(tubes.video_id);

xmld = struct([]);
v= 1;
for vv = 1 :  length(vids)
    
    action_indexes = find(strcmp(tubes.video_id,vids{vv}));
    videoName = vids{vv};
    xmld(v).videoName = videoName;
    actionscore = tubes.dpActionScore(action_indexes);
    path_scores = tubes.path_scores(1,action_indexes);
    
    ts = tubes.ts(action_indexes);
    te = tubes.te(action_indexes);
    act_nr = 1;
    
    for a = 1 : length(ts)
        act_ts = ts(a);
        act_te = te(a);
        act_dp_score = actionscore(a);
        act_path_scores = cell2mat(path_scores(a));
        
        %-----------------------------------------------------------
        act_scores = sort(act_path_scores(act_ts:act_te),'descend');
        %save('test.mat', 'act_scores'); pause;
        
        topk_mean = mean(act_scores(1:min(topk,length(act_scores))));
        
        bxs = tubes.path_boxes{action_indexes(a)}(act_ts:act_te,:);
        
        bxs = [bxs(:,1:2), bxs(:,3:4)-bxs(:,1:2)+1];
        
        label = tubes.label(action_indexes(a));
        
        % if topk_mean > kthresh && (act_te-act_ts) > min_num_frames && act_nr<=2
        % if topk_mean > kthresh && (act_te-act_ts) > min_num_frames
        if (act_te-act_ts) > min_num_frames
            xmld(v).score(act_nr) = topk_mean;
            xmld(v).nr(act_nr) = act_nr;
            xmld(v).class(act_nr) = label;
            xmld(v).framenr(act_nr).fnr = act_ts:act_te;
            
            xmld(v).boxes(act_nr).bxs = bxs;
            act_nr = act_nr+1;
        end
    end
    %fprintf('done %05d \n', v);
    v = v + 1;
end