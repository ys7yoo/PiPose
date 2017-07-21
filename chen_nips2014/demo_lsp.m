startup;
clear mex;
global GLOBAL_OVERRIDER;
GLOBAL_OVERRIDER = @lsp_conf;
conf = global_conf();
cachedir = conf.cachedir;
pa = conf.pa;
p_no = length(pa);
note = [conf.note];

% conf.useGpu = 0;        % DO NOT USE GPU => moved to global_conf() 
 
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
%tic
train_dcnn(pos_train, pos_val, neg_train, tsize, caffe_solver_file);
%disp('end of train_dcnn')
%toc

% RESULTS (in ../cache/lsp) :
% 
% lsp_iter_60000.caffemodel
% lsp_iter_60000.solverstate
%
% fully_conv_net_by_net_surgery.caffemodel

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% train graphical model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('training graphical model..................................................')
tic
model = train_model(note, pos_val, neg_val, tsize);
toc

% RESULTS (in ../cache/lsp)
% CNN_Deep_13_graphical_model.mat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('testing.................................................................')
tic
boxes = test_model([note,'_LSP'], model, pos_test);
toc

% RESULTS (in ../cache/lsp)
% CNN_Deep_13_LSP_raw_boxes.mat

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

xlabel('range');ylabel('percentage of detected joints');set(gcf,'paperposition', [0 0 6 5]);set(gcf,'papersize', [6 5]);saveas(gcf, 'lsp_pdf.pdf')

clear mex;
