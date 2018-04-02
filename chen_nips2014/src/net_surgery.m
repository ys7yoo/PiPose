function net_surgery(model_file, deploy_file, fully_conv_model_file, deploy_conv_file)
% ------------------------------------------------------------------------
% net_surgery_demo
% ------------------------------------------------------------------------

%deploy_file = './external/my_models/lsp/lsp_deploy.prototxt'
%model_file = '../cache/lsp/lsp_iter_60000.caffemodel'
if ~exist(deploy_file, 'file')
  error('cannot find %s (deploy_file)\n', deploy_file);
end
if ~exist(model_file, 'file')
  error('%s model has not been trained\n', model_file);
end

disp('loading deploy model') 
%% get weights in the net with fully-connected layers
% net = caffe.Net(deploy_file, model_file, 'test')
net = caffe.Net(deploy_file, 'test')
disp('loading trained weights')
net.copy_from(model_file); % load weights
disp trained('loaded')
disp(net.layer_names)


fc_weights = caffe('get_weights');
% print blob dimensions
fc_names = {'fc6', 'fc7', 'fc8'};
fc_layer_ids = zeros(3, 1);
for ii = 1:numel(fc_names)
  lname = fc_names{ii};
  for jj = 1:numel(fc_weights)
    if (strcmp(fc_weights(jj).layer_names, lname))
      fprintf('%s weights are ( %s) dimensional and biases are ( %s) dimensional\n', ...
        lname, sprintf('%d ', size(fc_weights(jj).weights{1})), ...
        sprintf('%d ', size(fc_weights(jj).weights{2})));
      fc_layer_ids(ii) = jj;
    end
  end
end

%% get weights in full-convolutional net
caffe('reset');
caffe('init', deploy_conv_file, model_file);
conv_weights = caffe('get_weights');
% print corresponding blob dimensions
conv_names = {'fc6-conv', 'fc7-conv', 'fc8-conv'};
conv_layer_ids = zeros(3, 1);
for ii = 1:numel(conv_names)
  lname = conv_names{ii};
  for jj = 1:numel(conv_weights)
    if (strcmp(conv_weights(jj).layer_names, lname))
      fprintf('%s weights are ( %s) dimensional and biases are ( %s) dimensional\n', ...
        lname, sprintf('%d ', size(conv_weights(jj).weights{1})), ...
        sprintf('%d ', size(conv_weights(jj).weights{2})));
      conv_layer_ids(ii) = jj;
    end
  end
end

%% tranplant paramters into full-convolutional net
trans_params = struct('weights', cell(numel(conv_names), 1), ...
  'layer_names', cell(numel(conv_names), 1));
for ii = 1:numel(conv_names)
  trans_params(ii).layer_names = conv_names{ii};
  weights = cell(2, 1);
  weights{1} = reshape(fc_weights(fc_layer_ids(ii)).weights{1}, size(conv_weights(conv_layer_ids(ii)).weights{1}));
  weights{2} = reshape(fc_weights(fc_layer_ids(ii)).weights{2}, size(conv_weights(conv_layer_ids(ii)).weights{2}));
  trans_params(ii).weights = weights;
end
caffe('set_weights', trans_params);
%% save
caffe('save', fully_conv_model_file);
caffe('reset');
