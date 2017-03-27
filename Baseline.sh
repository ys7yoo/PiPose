CAFFE_ROOT=~/src/caffe-rc5
#CAFFE_ROOT=~/src/caffe-rc3
#CAFFE_ROOT=./caffe-multi

#remove gpu option!
# also changed solver.prototxt

GLOG_logtostderr=1 ${CAFFE_ROOT}/build/tools/caffe train \
-solver ./protofiles/solver.prototxt \
-weights cache/VGG_ILSVRC_16_layers_full_conv.caffemodel \
2>&1 | tee log_test.txt
