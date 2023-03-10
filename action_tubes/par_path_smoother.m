% -------------------------------------------------------------------
% Author: Gurkirt Singh
% It applies a second pass dynamic programming on the action paths for 
% temporal label smoothing and trimming to generate the final action tubes.
% -------------------------------------------------------------------
function final_tubes = par_path_smoother(tubes, alpha, num_action)

final_tubes = struct('ts',[],'te',[],'label',[],'path_total_score',[],...
    'dpActionScore',[],'dpPathScore',[],'vid',[],...
    'path_boxes',cell(1),'path_scores',cell(1),'video_id',cell(1));


alltubes  = cell(length(tubes),1);

% make it parfor in the final version
parfor t = 1 : length(tubes)
    %     fprintf('[%03d/%03d] calpha %04d\n',t,length(tubes),uint16(calpha*100));
    %     fprintf('.');
    video_id = tubes(t).video_id;
    %     fprintf('[doing for %s %d out of %d]\n',video_id,t,length(tubes));
    alltubes{t} = actionPathSmoother4oneVideo(tubes(t).paths, alpha, num_action, video_id) ;
end

action_count = 1;
for t = 1 : length(tubes)
    vid_tubes = alltubes{t};
    for  k=1:length(vid_tubes.ts)
        final_tubes.ts(action_count) = vid_tubes.ts(k);
        final_tubes.video_id{action_count} = vid_tubes.video_id{k};
        final_tubes.te(action_count) = vid_tubes.te(k);
        final_tubes.dpActionScore(action_count) = vid_tubes.dpActionScore(k);
        final_tubes.label(action_count) = vid_tubes.label(k);
        final_tubes.dpPathScore(action_count) = vid_tubes.dpPathScore(k);
        final_tubes.path_total_score(action_count) = vid_tubes.path_total_score(k);
        final_tubes.path_boxes{action_count} = vid_tubes.path_boxes{k};
        final_tubes.path_scores{action_count} = vid_tubes.path_scores{k};
        action_count = action_count + 1;
    end
    
end
end

function final_tubes = actionPathSmoother4oneVideo(video_paths, alpha, num_action, video_id)
action_count =1;
final_tubes = struct('ts',[],'te',[],'label',[],'path_total_score',[],...
    'dpActionScore',[],'dpPathScore',[],'vid',[],...
    'path_boxes',cell(1),'path_scores',cell(1),'video_id',cell(1));

if ~isempty(video_paths)
    %gt_ind = find(strcmp(video_id,annot.videoName));
    %number_frames = length(video_paths{1}(1).idx);
    
    for a = 1 :num_action
        if 1 %a == 9
            action_paths = video_paths{a};
            for p = 1 : min(3,length(action_paths)) % taking top 3 paths
                % M = action_paths(p).allScore'; % original
                M = action_paths(p).allScore(:,1:2)';
                % M = M +20;
                
                % [pred_path,time,D] = dpEM_max(M,alpha(a));
                [pred_path,time,D] = dpEM_max(M,alpha(a));
                [ Ts, Te, Scores,Label, DpPathScore] = extract_action(pred_path,time,D,a);
                for k = 1 : length(Ts)
                    final_tubes.ts(action_count) = Ts(k);
                    final_tubes.video_id{action_count} = video_id;
                    %                         final_tubes.vid(action_count) = vid_num;
                    final_tubes.te(action_count) = Te(k);
                    final_tubes.dpActionScore(action_count) = Scores(k);
                    final_tubes.label(action_count) = Label(k)*a;
                    final_tubes.dpPathScore(action_count) = DpPathScore(k);
                    final_tubes.path_total_score(action_count) = action_paths(p).total_score;
                    final_tubes.path_boxes{action_count} = action_paths(p).boxes;
                    final_tubes.path_scores{action_count} = action_paths(p).scores;
                    action_count = action_count + 1;
                end
                
            end
        end
    end
    
end
%fprintf('done %s \n', video_id);
end

function [ts,te,scores,label,total_score] = extract_action(p,q,D,action)
action = 1;
indexs = find(p==action);

if isempty(indexs)
    ts = []; te = []; scores = []; label = []; total_score = [];
    
else
    indexs_diff = [indexs,indexs(end)+1] - [indexs(1)-2,indexs];
    ts = find(indexs_diff>1);
    
    if length(ts)>1
        te = [ts(2:end)-1,length(indexs)];
    else
        te = length(indexs);
    end
    ts = indexs(ts);
    te = indexs(te);
    scores = (D(action,q(te)) - D(action,q(ts)))./(te-ts);
    label = ones(length(ts),1)*action;
    total_score = ones(length(ts),1)*D(p(end),q(end))/length(p);
end
end

function [ts,te,scores,label,total_score] = extract_action_original(p,q,D,action)
% p(1:1) = 1;
indexs = find(p==action);

if isempty(indexs)
    ts = []; te = []; scores = []; label = []; total_score = [];
    
else
    indexs_diff = [indexs,indexs(end)+1] - [indexs(1)-2,indexs];
    ts = find(indexs_diff>1);
    
    if length(ts)>1
        te = [ts(2:end)-1,length(indexs)];
    else
        te = length(indexs);
    end
    ts = indexs(ts);
    te = indexs(te);
    scores = (D(action,q(te)) - D(action,q(ts)))./(te-ts);
    label = ones(length(ts),1)*action;
    total_score = ones(length(ts),1)*D(p(end),q(end))/length(p);
end
end
