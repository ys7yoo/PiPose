startup;
clear mex;
global GLOBAL_OVERRIDER;
GLOBAL_OVERRIDER = @lsp_conf;
conf = global_conf();
cachedir = conf.cachedir;
pa = conf.pa;
p_no = length(pa);
note = [conf.note];

conf.useGpu = 0;        % DO NOT USE GPU
 
diary([cachedir note '_log_' datestr(now,'mm-dd-yy') '.txt']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% read data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('read LSP data')
tic
[pos_train, pos_val, pos_test, neg_train, neg_val, tsize] = LSP_data();
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% train dcnn
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if conf.useGpu
    caffe_solver_file = 'external/my_models/lsp/lsp_solver.prototxt';
else
    caffe_solver_file = 'external/my_models/lsp/lsp_solver_cpu.prototxt';
end
tic
train_dcnn(pos_train, pos_val, neg_train, tsize, caffe_solver_file);
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% train graphical model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
model = train_model(note, pos_val, neg_val, tsize);
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
boxes = test_model([note,'_LSP'], model, pos_test);
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% evaluation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eval_method = {'strict_pcp', 'pdj'};
fprintf('============= On test =============\n');
ests = conf.box2det(boxes, p_no);
% generate part stick from joints locations
for ii = 1:numel(ests)
    ests(ii).sticks = conf.joint2stick(ests(ii).joints);
    pos_test(ii).sticks = conf.joint2stick(pos_test(ii).joints);
end
show_eval(pos_test, ests, conf, eval_method);
diary off;
clear mex;
