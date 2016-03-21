#!/usr/bin/env sh

DATA=./

$TOOLS/compute_image_mean $DATA/train_rescaled_data_lmdb \
  $DATA/train_data_mean.binaryproto

$TOOLS/compute_image_mean $DATA/val_rescaled_data_lmdb \
  $DATA/val_data_mean.binaryproto

echo "Done."
