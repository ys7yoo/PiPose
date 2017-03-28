#!/bin/bash

## 
/opt/matlab/bin/mex -O -outdir bin  src/mex/distance_transform.cpp

/opt/matlab/bin/mex -O -outdir bin  external/qpsolver/qp_one_sparse.cc

/opt/matlab/bin/mex -O -outdir bin  external/qpsolver/score.cc

/opt/matlab/bin/mex -O -outdir bin  external/qpsolver/lincomb.cc

##
CAFFE_ROOT="./external/caffe"

/opt/matlab/bin/mex -O -outdir bin  -I$CAFFE_ROOT$/build/src -I/opt/local/include -lprotobuf -llmdb /build/lib/libcaffe.a src/mex/store_patch.cpp 

