
To create databases:

- Launch create_dataset.py : this creates two .txt files allowing to create two databases. 
One for training and one for validation. Files contain path to image, labels.
- Launch create_lmdb_train.py. This creates 4 databases for training from the previous txt file.
One with images, one with x label, one with y label, one with z label.
- Launch create_lmdb_val.py. This creates the database for validation from the previous txt file.
One with images, one with x label, one with y label, one with z label.
