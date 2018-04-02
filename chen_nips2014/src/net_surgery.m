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
%            layer_vec: [1×23 caffe.Layer]
%             blob_vec: [1×14 caffe.Blob]
%               inputs: {'data'}
%              outputs: {'prob'}
%     name2layer_index: [23×1 containers.Map]
%      name2blob_index: [14×1 containers.Map]
%          layer_names: {23×1 cell}
%           blob_names: {14×1 cell}
disp('loading trained weights')
net.copy_from(model_file); % load weights
disp trained('loaded')
disp(net.layer_names)


%fc_weights = caffe('get_weights');
% print blob dimensions
fc_names = {'fc6', 'fc7', 'fc8'};
fc_layer_ids = zeros(3, 1);
for ii = 1:numel(fc_names)
    lname = fc_names{ii};
    layer_index=net.name2layer_index(lname);
    fc_layer_ids(ii) = layer_index;

%     for jj = 1:numel(fc_weights)
%         if (strcmp(fc_weights(jj).layer_names, lname))
%             fprintf('%s weights are ( %s) dimensional and biases are ( %s) dimensional\n', ...
%                 lname, sprintf('%d ', size(fc_weights(jj).weights{1})), ...
%                 sprintf('%d ', size(fc_weights(jj).weights{2})));
%             fc_layer_ids(ii) = jj;
%         end
%     end
end

% w=net.params('fc6',1).get_data();

    
%% get weights in full-convolutional net
%caffe('reset');
net2=caffe.Net(deploy_conv_file, model_file,'test');
% deploy_conv_file = './external/my_models/lsp/lsp_deploy_conv.prototxt'
% model_file = '../cache/lsp/lsp_iter_60000.caffemodel'    
%            layer_vec: [1×23 caffe.Layer]
%             blob_vec: [1×14 caffe.Blob]
%               inputs: {'data'}
%              outputs: {'prob'}
%     name2layer_index: [23×1 containers.Map]
%      name2blob_index: [14×1 containers.Map]
%          layer_names: {23×1 cell}
%           blob_names: {14×1 cell}
%conv_weights = caffe('get_weights');
% print corresponding blob dimensions
conv_names = {'fc6-conv', 'fc7-conv', 'fc8-conv'};
conv_layer_ids = zeros(3, 1);
for ii = 1:numel(conv_names)
    lname = conv_names{ii};
    layer_index=net2.name2layer_index(lname);
    conv_layer_ids(ii) = layer_index;
end
% for ii = 1:numel(conv_names)
%     lname = conv_names{ii};
%     for jj = 1:numel(conv_weights)
%         if (strcmp(conv_weights(jj).layer_names, lname))
%             fprintf('%s weights are ( %s) dimensional and biases are ( %s) dimensional\n', ...
%                 lname, sprintf('%d ', size(conv_weights(jj).weights{1})), ...
%                 sprintf('%d ', size(conv_weights(jj).weights{2})));
%             conv_layer_ids(ii) = jj;
%         end
%     end
% end

%% tranplant paramters into full-convolutional net

% test code
for ii = 1:numel(fc_names)
    sz=size(net2.params(conv_names{ii},1).get_data());
    net2.params(conv_names{ii},1).set_data(reshape(net.params(fc_names{ii},1).get_data(), sz));
    sz=size(net2.params(conv_names{ii},2).get_data());
    net2.params(conv_names{ii},2).set_data(reshape(net.params(fc_names{ii},2).get_data(), sz));
end

net2.save(fully_conv_model_file)
