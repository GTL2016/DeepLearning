#!/usr/bin/env sh

DATA=./

$TOOLS/compute_image_mean $DATA/val_rescaled_data_lmdb \
  $DATA/val_data_mean.binaryproto

echo "Val dataset done."

$TOOLS/compute_image_mean $DATA/train_rescaled_data_lmdb \
  $DATA/train_data_mean.binaryproto

echo "Train dataset done."
