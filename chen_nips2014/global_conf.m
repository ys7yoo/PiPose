function conf = global_conf()

assert_not_in_parallel_worker();

%% dataset
conf.interval = 10; % 10 levels from 1 to 1/2
conf.memsize = 0.5; % 0.5 gb
conf.NEG_N = 80;
conf.device_id = 0;
% conf.caffe_root = './external/caffe';
conf.caffe_root = [getenv('HOME') '/src/caffe']
% default configurations
conf.mining_neg = true;
conf.mining_pos = false;
conf.K = 13;
conf.test_with_detection = false;

conf.useGpu = 1;
% conf.useGpu = 0;
% conf.batch_size = 1024;
conf.batch_size = 4096;

conf.at_least_one = true;

%% override some configurations
global GLOBAL_OVERRIDER;

if ~isempty(GLOBAL_OVERRIDER)
    % modify some fields of conf
    conf = GLOBAL_OVERRIDER(conf);
    % ======= fields constructed from existing configurations =======
    conf.note = ['CNN_Deep_', num2str(conf.K)];
    conf.cachedir = ['./cache/', conf.dataset, '/'];
    %conf.cachedir = ['../cache/', conf.dataset, '/'];  % changed on 2017. 4. 22
    if ~exist(conf.cachedir, 'dir')
        mkdir(conf.cachedir);
    end
    conf.cnn.image_mean_file = [conf.cachedir, conf.dataset, '_mean.mat'];
end

function assert_not_in_parallel_worker()
%%
% Matlab does not support accessing global variables from
% parallel workers. The result of reading a global is undefined
% and in practice has odd and inconsistent behavoir.
% The configuraton override mechanism relies on a global
% variable. To avoid hard-to-find bugs, we make sure that
% global_conf cannot be called from a parallel worker.

t = [];
if usejava('jvm')
    try
        t = getCurrentTask();
    catch
    end
end

if ~isempty(t)
    msg = 'global_conf() cannot be called from a parallel worker ';
    error(msg);
end
