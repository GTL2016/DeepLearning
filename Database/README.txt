
To create databases:

- Launch best_classes_generation.py with "gtl" or "supelec" option
- Launch create_dataset.py with "gtl" or "supelec" option
- Launch compute_scale.py
- Launch compute_means.py
- Launch scale_dataset.py
- Launch create_lmdb.py with "train_rescaled" or "val_rescaled" and 240 352 (new size of images)
- Launch make_pfe_mean.sh
