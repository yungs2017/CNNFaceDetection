#!/usr/bin/env sh
export PYTHONPATH=$PYTHONPATH:../common/pythonLayer
set -e
LOG=models/train.log
/caffe-master/build/install/bin/caffe train \
          --solver=./solver.prototxt -gpu 0 \
         2>&1 | tee $LOG

