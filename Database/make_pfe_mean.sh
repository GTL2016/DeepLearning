#!/usr/bin/env sh

DATA=/home/yr/Documents/Project/
TOOLS=/home/yr/Programmes/caffe-master/build/tools

$TOOLS/compute_image_mean $DATA/train_data_lmdb \
  $DATA/train_mean.binaryproto

$TOOLS/compute_image_mean $DATA/val_data_lmdb \
  $DATA/val_mean.binaryproto
echo "Done."
