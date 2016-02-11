#!/usr/bin/env sh

DATA=./
TOOLS=$CAFFE_ROOT/../build/tools

$TOOLS/compute_image_mean $DATA/train_data_lmdb \
  $DATA/train_mean.binaryproto

$TOOLS/compute_image_mean $DATA/val_data_lmdb \
  $DATA/val_mean.binaryproto
echo "Done."
