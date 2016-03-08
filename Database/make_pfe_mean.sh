#!/usr/bin/env sh

DATA=./
TOOLS=$CAFFE_ROOT/../tools

$TOOLS/compute_image_mean $DATA/train_data_lmdb \
  $DATA/train_data_mean.binaryproto

$TOOLS/compute_image_mean $DATA/val_data_lmdb \
  $DATA/val_data_mean.binaryproto

$TOOLS/compute_image_mean $DATA/train_label_lmdb \
  $DATA/train_label_mean.binaryproto

$TOOLS/compute_image_mean $DATA/val_label_lmdb \
  $DATA/val_label_mean.binaryproto

echo "Done."
