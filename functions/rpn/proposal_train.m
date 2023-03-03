function save_model_path = proposal_train(conf, imdb_train, roidb_train, output_dir, restore, solver_state_file, loss_log_file_path, type, varargin)
% inputs
ip = inputParser;
ip.addRequired('conf',                                      @isstruct);
ip.addRequired('imdb_train',                                @iscell);
ip.addRequired('roidb_train',                               @iscell);
ip.addRequired('output_dir',                                @isstr);
ip.addRequired('restore',                                   @isscalar);
ip.addRequired('solver_state_file',                         @isstr);
ip.addRequired('loss_log_file_path',                        @isstr);
ip.addRequired('type',                                      @isstr);



%######## the following entries will be overidden by the passes
%parameters
ip.addParamValue('do_val',              false,              @isscalar);
ip.addParamValue('imdb_val',            struct(),           @isstruct);
ip.addParamValue('roidb_val',           struct(),           @isstruct);



% #########  uncomment ################
ip.addParamValue('val_iters',           50,                @isscalar); % ori - 500
ip.addParamValue('val_interval',        50,               @isscalar); % ori - 2000 - do validation after these many iterations
ip.addParamValue('snapshot_interval',  100,              @isscalar); % 10000


% Max pixel size of a scaled input image
ip.addParamValue('solver_def_file',     fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'solver.prototxt'), ...
    @isstr);
ip.addParamValue('net_file',            fullfile(pwd, 'proposal_models', 'Zeiler_conv5', 'Zeiler_conv5.caffemodel'), ...
    @isstr);
ip.addParamValue('cache_name',          'Zeiler_conv5', ...
    @isstr);

ip.parse(conf, imdb_train, roidb_train, output_dir, restore, solver_state_file, loss_log_file_path, type, varargin{:});
opts = ip.Results;

%%- try to find trained model
imdbs_name = cell2mat(cellfun(@(x) x.name, imdb_train, 'UniformOutput', false));
num_classes = cell2mat(cellfun(@(x) length(x.classes), imdb_train, 'UniformOutput', false));
num_imgs = cell2mat(cellfun(@(x) length(x.image_ids), imdb_train, 'UniformOutput', false));
num_imgs_val = cell2mat(cellfun(@(x) length(x.image_ids), {opts.imdb_val}, 'UniformOutput', false));

cache_dir = fullfile(opts.output_dir, ['rpn_trained_model_' opts.type], imdbs_name);
save_model_path = fullfile(cache_dir, 'final');

if exist(save_model_path, 'file')
    return;
end

loss_log_file = [opts.loss_log_file_path '/' 'loss_log_rpn_' type '.txt'];
train_err_file = [opts.loss_log_file_path '/' 'train_err_rpn_' type '.txt'];
vald_err_file = [opts.loss_log_file_path '/' 'validation_err_rpn_' type '.txt'];

loss_log_fileID = fopen(loss_log_file, 'w');
train_err_fileID = fopen(train_err_file, 'w');
vald_err_fileID = fopen(vald_err_file, 'w');

imdbs_name = cell2mat(cellfun(@(x) x.name, imdb_train, 'UniformOutput', false));
%cache_dir = fullfile(opts.output_dir, 'output', 'rpn_cachedir', opts.cache_name, imdbs_name);
%mkdir_if_missing(cache_dir);



%%- init
% init caffe solver
caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
caffe.init_log(caffe_log_file_base);
caffe_solver = caffe.Solver(opts.solver_def_file);

%%--- EITHER RESTORE FROM THE CAHCED SOLVER STATE ----
%%--- OR INITIALISED WITH THE PRETRAINED MODEL WEIGHTS ---
if restore
    caffe_solver.restore(opts.solver_state_file);
else
    caffe_solver.net.copy_from(opts.net_file);
end

% init log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
mkdir_if_missing(fullfile(cache_dir, 'log'));
log_file = fullfile(cache_dir, 'log', ['train_', timestamp, '.txt']);
diary(log_file);

% set random seed
prev_rng = seed_rand(conf.rng_seed);
caffe.set_random_seed(conf.rng_seed);

% set gpu/cpu
if conf.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end

disp('conf:');
disp(conf);
disp('opts:');
disp(opts);

%----training data ---
image_roidb_base_path = [opts.output_dir '/' 'rpn_prep_img_roidb_' type '/' 'train']; %'/' 'rpn_proposal_prepare_image_roidb_v2' '/' 'train']; 
image_roidb_file_path = [image_roidb_base_path '/' 'image_roidb_final'];
bmst = load([image_roidb_base_path '/' 'bbox_means.mat']);
bbox_means = bmst.bbox_means; clear bmst;
bsst = load([image_roidb_base_path '/' 'bbox_stds.mat']);
bbox_stds = bsst.bbox_stds; clear bsst;

%--- validation data ---
image_roidb_base_path_val = [opts.output_dir '/' 'rpn_prep_img_roidb_' type '/' 'validation']; %'/' 'rpn_proposal_prepare_image_roidb_v2' '/' 'validation'];
image_roidb_file_path_val = [image_roidb_base_path_val '/' 'image_roidb_final'];

fprintf('Done.\n');

if opts.do_val    
   % [image_roidb_val] = proposal_prepare_image_roidb(conf, opts.imdb_val, opts.roidb_val, bbox_means, bbox_stds);
   % fprintf('Done.\n');
   
    % fix validation data
    % shuffled_inds_val   = generate_random_minibatch_original([], image_roidb_val, conf.ims_per_batch);
    shuffled_inds_val   = generate_random_minibatch([], num_imgs_val, conf.ims_per_batch);
    shuffled_inds_val   = shuffled_inds_val(randperm(length(shuffled_inds_val), opts.val_iters));
end

conf.classes = opts.imdb_train{1}.classes;

%%-  try to train/val with images which have maximum size potentially, to validate whether the gpu memory is enough
check_gpu_memory(conf, caffe_solver, opts.do_val);

%%- -------------------- Training --------------------

proposal_generate_minibatch_fun = @proposal_generate_minibatch;
visual_debug_fun                = @proposal_visual_debug;

% training
shuffled_inds = [];
train_results = [];
val_results = [];
iter_ = caffe_solver.iter();
max_iter = caffe_solver.max_iter();

while (iter_ < max_iter)  
    
    caffe_solver.net.set_phase('train');
    
    % generate minibatch training data
    [shuffled_inds, sub_db_inds] = generate_random_minibatch(shuffled_inds, num_imgs, conf.ims_per_batch);
    
        
    for i=1:length(sub_db_inds)
        irt_file = sprintf('%s/%06d.mat', image_roidb_file_path, sub_db_inds(i)); % irt-image_roidb_train
        irt_st = load(irt_file);
        image_roidb_train(i,1) = irt_st.image_roidb_st_2; clear irt_st;
    end
    
     if ~mod(iter_-1, opts.snapshot_interval)
        ts = tic;
        fprintf('ts init at iteration:= %d\n', iter_);
        fprintf('reading %d image per iteration : image_roidb file : \n %s \n', length(sub_db_inds), irt_file);
    end
    
    %[net_inputs, scale_inds] = proposal_generate_minibatch_fun(conf, image_roidb_train(sub_db_inds));
    [net_inputs, scale_inds] = proposal_generate_minibatch_fun(conf, image_roidb_train);
    
    % visual_debug_fun(conf, image_roidb_train(sub_db_inds), net_inputs, bbox_means, bbox_stds, conf.classes, scale_inds);
    caffe_solver.net.reshape_as_input(net_inputs);
    
    % one iter SGD update
    caffe_solver.net.set_input_data(net_inputs);
    caffe_solver.step(1);
    rst = caffe_solver.net.get_output();
    rst = check_error(rst, caffe_solver);
    train_results = parse_rst(train_results, rst);
    
    
        
    % do valdiation per val_interval iterations
    if ~mod(iter_, opts.val_interval)
        if opts.do_val
            val_results = do_validation(conf, caffe_solver, proposal_generate_minibatch_fun, shuffled_inds_val, image_roidb_file_path_val);
            % val_results = do_validation_original(conf, caffe_solver, proposal_generate_minibatch_fun, image_roidb_val, shuffled_inds_val);
        end
        
        check_loss(rst, caffe_solver, net_inputs, loss_log_fileID);
        show_state(iter_, train_results, val_results, train_err_fileID, vald_err_fileID);
        train_results = [];
        diary; diary; % flush diary
    end
    
    % snapshot
    if ~mod(iter_, opts.snapshot_interval)
        snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
        te = toc(ts);
        fprintf('time taken to complete %d iteration:= %.3fs\n', opts.snapshot_interval, te);
    end
    
     
    
    iter_ = caffe_solver.iter();
end

% final validation
if opts.do_val
    do_validation(conf, caffe_solver, proposal_generate_minibatch_fun, shuffled_inds_val, image_roidb_file_path_val);
    %do_validation(conf, caffe_solver, proposal_generate_minibatch_fun, image_roidb_val, shuffled_inds_val);
end
% final snapshot
snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, sprintf('iter_%d', iter_));
save_model_path = snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, 'final');

diary off;
fclose(loss_log_fileID);
fclose(train_err_fileID);
fclose(vald_err_fileID);


caffe.reset_all();
rng(prev_rng);

end

function val_results = do_validation(conf, caffe_solver, proposal_generate_minibatch_fun, shuffled_inds_val, image_roidb_file_path_val)
val_results = [];

caffe_solver.net.set_phase('test');
for i = 1:length(shuffled_inds_val)
    sub_db_inds = shuffled_inds_val{i};

 for j=1:length(sub_db_inds)
        irv_file = sprintf('%s/%06d.mat', image_roidb_file_path_val, sub_db_inds(j)); % irv-image_roidb_validation
        irv_st = load(irv_file);
        image_roidb_val(j,1) = irv_st.image_roidb_st_2; clear irv_st;
    end

    [net_inputs, ~] = proposal_generate_minibatch_fun(conf, image_roidb_val);
    
    % Reshape net's input blobs
    caffe_solver.net.reshape_as_input(net_inputs);
    
    caffe_solver.net.forward(net_inputs);
    rst = caffe_solver.net.get_output();
    rst = check_error(rst, caffe_solver);
    val_results = parse_rst(val_results, rst);
end
end

function val_results = do_validation_original(conf, caffe_solver, proposal_generate_minibatch_fun, image_roidb_val, shuffled_inds_val)
val_results = [];

caffe_solver.net.set_phase('test');
for i = 1:length(shuffled_inds_val)
    sub_db_inds = shuffled_inds_val{i};
    [net_inputs, ~] = proposal_generate_minibatch_fun(conf, image_roidb_val(sub_db_inds));
    
    % Reshape net's input blobs
    caffe_solver.net.reshape_as_input(net_inputs);
    
    caffe_solver.net.forward(net_inputs);
    rst = caffe_solver.net.get_output();
    rst = check_error(rst, caffe_solver);
    val_results = parse_rst(val_results, rst);
end
end

%--- experimental purpose remove this later
function [shuffled_inds, sub_inds] = generate_random_minibatch_original(shuffled_inds, image_roidb_train, ims_per_batch)

    % shuffle training data per batch
    if isempty(shuffled_inds)
        % make sure each minibatch, only has horizontal images or vertical
        % images, to save gpu memory
        
        hori_image_inds = arrayfun(@(x) x.im_size(2) >= x.im_size(1), image_roidb_train, 'UniformOutput', true);
        vert_image_inds = ~hori_image_inds;
        hori_image_inds = find(hori_image_inds);
        vert_image_inds = find(vert_image_inds);
        
        % random perm
        lim = floor(length(hori_image_inds) / ims_per_batch) * ims_per_batch;
        hori_image_inds = hori_image_inds(randperm(length(hori_image_inds), lim));
        lim = floor(length(vert_image_inds) / ims_per_batch) * ims_per_batch;
        vert_image_inds = vert_image_inds(randperm(length(vert_image_inds), lim));
        
        % combine sample for each ims_per_batch 
        hori_image_inds = reshape(hori_image_inds, ims_per_batch, []);
        vert_image_inds = reshape(vert_image_inds, ims_per_batch, []);
        
        shuffled_inds = [hori_image_inds, vert_image_inds];
        shuffled_inds = shuffled_inds(:, randperm(size(shuffled_inds, 2)));
        
        shuffled_inds = num2cell(shuffled_inds, 1);
    end
    
    if nargout > 1
        % generate minibatch training data
        sub_inds = shuffled_inds{1};
        assert(length(sub_inds) == ims_per_batch);
        shuffled_inds(1) = [];
    end
end

function [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, num_imgs, ims_per_batch)

% shuffle training data per batch
if isempty(shuffled_inds)
    % make sure each minibatch, only has horizontal images or vertical
    % images, to save gpu memory
    
    %hori_image_inds = arrayfun(@(x) x.im_size(2) >= x.im_size(1), image_roidb_train, 'UniformOutput', true);
     hori_image_ids = ones(num_imgs,1);
    hori_image_inds = hori_image_ids==1;
    vert_image_inds = ~hori_image_inds;
    hori_image_inds = find(hori_image_inds);
    vert_image_inds = find(vert_image_inds);
    
    % random perm
    lim = floor(length(hori_image_inds) / ims_per_batch) * ims_per_batch;
    hori_image_inds = hori_image_inds(randperm(length(hori_image_inds), lim));
    lim = floor(length(vert_image_inds) / ims_per_batch) * ims_per_batch;
    vert_image_inds = vert_image_inds(randperm(length(vert_image_inds), lim));
    
    % combine sample for each ims_per_batch
    hori_image_inds = reshape(hori_image_inds, ims_per_batch, []);
    vert_image_inds = reshape(vert_image_inds, ims_per_batch, []);
    
    shuffled_inds = [hori_image_inds, vert_image_inds];
    shuffled_inds = shuffled_inds(:, randperm(size(shuffled_inds, 2)));
    
    shuffled_inds = num2cell(shuffled_inds, 1);
end

if nargout > 1
    % generate minibatch training data
    sub_inds = shuffled_inds{1};
    assert(length(sub_inds) == ims_per_batch);
    shuffled_inds(1) = [];
end
end

function rst = check_error(rst, caffe_solver)

cls_score = caffe_solver.net.blobs('proposal_cls_score_reshape').get_data();
labels = caffe_solver.net.blobs('labels_reshape').get_data();
labels_weights = caffe_solver.net.blobs('labels_weights_reshape').get_data();

accurate_fg = (cls_score(:, :, 2) > cls_score(:, :, 1)) & (labels == 1);
accurate_bg = (cls_score(:, :, 2) <= cls_score(:, :, 1)) & (labels == 0);
accurate = accurate_fg | accurate_bg;
accuracy_fg = sum(accurate_fg(:) .* labels_weights(:)) / (sum(labels_weights(labels == 1)) + eps);
accuracy_bg = sum(accurate_bg(:) .* labels_weights(:)) / (sum(labels_weights(labels == 0)) + eps);

rst(end+1) = struct('blob_name', 'accuracy_fg', 'data', accuracy_fg);
rst(end+1) = struct('blob_name', 'accuracy_bg', 'data', accuracy_bg);
end

function check_gpu_memory(conf, caffe_solver, do_val)
%%-  try to train/val with images which have maximum size potentially, to validate whether the gpu memory is enough

% generate pseudo training data with max size
im_blob = single(zeros(max(conf.scales), conf.max_size, 3, conf.ims_per_batch));

anchor_num = size(conf.anchors, 1);
output_width = conf.output_width_map.values({size(im_blob, 1)});
output_width = output_width{1};
output_height = conf.output_width_map.values({size(im_blob, 2)});
output_height = output_height{1};
labels_blob = single(zeros(output_width, output_height, anchor_num, conf.ims_per_batch));
labels_weights = labels_blob;
bbox_targets_blob = single(zeros(output_width, output_height, anchor_num*4, conf.ims_per_batch));
bbox_loss_weights_blob = bbox_targets_blob;

net_inputs = {im_blob, labels_blob, labels_weights, bbox_targets_blob, bbox_loss_weights_blob};

% Reshape net's input blobs
caffe_solver.net.reshape_as_input(net_inputs);

% one iter SGD update
caffe_solver.net.set_input_data(net_inputs);
caffe_solver.step(1);

if do_val
    % use the same net with train to save memory
    caffe_solver.net.set_phase('test');
    caffe_solver.net.forward(net_inputs);
    caffe_solver.net.set_phase('train');
end
end

function model_path = snapshot(conf, caffe_solver, bbox_means, bbox_stds, cache_dir, file_name)
anchor_size = size(conf.anchors, 1);
bbox_stds_flatten = repmat(reshape(bbox_stds', [], 1), anchor_size, 1);
bbox_means_flatten = repmat(reshape(bbox_means', [], 1), anchor_size, 1);

% merge bbox_means, bbox_stds into the model
bbox_pred_layer_name = 'proposal_bbox_pred';
weights = caffe_solver.net.params(bbox_pred_layer_name, 1).get_data();
biase = caffe_solver.net.params(bbox_pred_layer_name, 2).get_data();
weights_back = weights;
biase_back = biase;

weights = ...
    bsxfun(@times, weights, permute(bbox_stds_flatten, [2, 3, 4, 1])); % weights = weights * stds;
biase = ...
    biase .* bbox_stds_flatten + bbox_means_flatten; % bias = bias * stds + means;

caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights);
caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase);

model_path = fullfile(cache_dir, file_name);
caffe_solver.net.save(model_path);
fprintf('Saved as %s\n', model_path);

% restore net to original state
caffe_solver.net.set_params_data(bbox_pred_layer_name, 1, weights_back);
caffe_solver.net.set_params_data(bbox_pred_layer_name, 2, biase_back);

end

function show_state(iter, train_results, val_results, train_err_fileID, vald_err_fileID)
fprintf('\n------------------------- Iteration %d -------------------------\n', iter);
fprintf('Training : err_fg %.3g, err_bg %.3g, loss (cls %.3g + reg %.3g)\n', ...
    1 - mean(train_results.accuracy_fg.data), 1 - mean(train_results.accuracy_bg.data), ...
    mean(train_results.loss_cls.data), ...
    mean(train_results.loss_bbox.data));

fprintf(train_err_fileID, '%f %f %f %f\n', ...
    1 - mean(train_results.accuracy_fg.data), ...
    1 - mean(train_results.accuracy_bg.data), ...
    mean(train_results.loss_cls.data), ...
    mean(train_results.loss_bbox.data));

if exist('val_results', 'var') && ~isempty(val_results)
    fprintf('Testing  : err_fg %.3g, err_bg %.3g, loss (cls %.3g + reg %.3g)\n', ...
        1 - mean(val_results.accuracy_fg.data), 1 - mean(val_results.accuracy_bg.data), ...
        mean(val_results.loss_cls.data), ...
        mean(val_results.loss_bbox.data));
    
    fprintf(vald_err_fileID, '%f %f %f %f\n', ...
         1 - mean(val_results.accuracy_fg.data),...
         1 - mean(val_results.accuracy_bg.data), ...
        mean(val_results.loss_cls.data), ...
        mean(val_results.loss_bbox.data));
        
end
end

function check_loss(rst, caffe_solver, input_blobs, loss_log_fileID)
im_blob = input_blobs{1};
labels_blob = input_blobs{2};
label_weights_blob = input_blobs{3};
bbox_targets_blob = input_blobs{4};
bbox_loss_weights_blob = input_blobs{5};

regression_output = caffe_solver.net.blobs('proposal_bbox_pred').get_data();
% smooth l1 loss
regression_delta = abs(regression_output(:) - bbox_targets_blob(:));
regression_delta_l2 = regression_delta < 1;
regression_delta = 0.5 * regression_delta .* regression_delta .* regression_delta_l2 + (regression_delta - 0.5) .* ~regression_delta_l2;
regression_loss = sum(regression_delta.* bbox_loss_weights_blob(:)) / size(regression_output, 1) / size(regression_output, 2);

confidence = caffe_solver.net.blobs('proposal_cls_score_reshape').get_data();
labels = reshape(labels_blob, size(labels_blob, 1), []);
label_weights = reshape(label_weights_blob, size(label_weights_blob, 1), []);

confidence_softmax = bsxfun(@rdivide, exp(confidence), sum(exp(confidence), 3));
confidence_softmax = reshape(confidence_softmax, [], 2);
confidence_loss = confidence_softmax(sub2ind(size(confidence_softmax), 1:size(confidence_softmax, 1), labels(:)' + 1));
confidence_loss = -log(confidence_loss);
confidence_loss = sum(confidence_loss' .* label_weights(:)) / sum(label_weights(:));

results = parse_rst([], rst);
% fprintf('C++   : conf %f, reg %f\n', results.loss_cls.data, results.loss_bbox.data);
% fprintf('Matlab: conf %f, reg %f\n', confidence_loss, regression_loss);

fprintf(loss_log_fileID, '%f %f\n', confidence_loss, regression_loss);

end