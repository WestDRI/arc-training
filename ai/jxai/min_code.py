import os
import polars as pl
import imageio.v3 as iio
import grain.python as grain
import numpy as np
from datasets import Dataset

base_dir = 'nabirds'
cleaned_img_dir = os.path.join(base_dir, 'cleaned_images')

metadata = pl.read_parquet('metadata.parquet')
metadata_train = metadata.filter(pl.col("is_training_img") == 1)
metadata_val = metadata.filter(pl.col("is_training_img") == 0)

class NABirdsDataset:
    """NABirds dataset class."""
    def __init__(self, metadata_file, data_dir):
        self.metadata_file = metadata_file
        self.data_dir = data_dir
    def __len__(self):
        return len(self.metadata_file)
    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.metadata_file.get_column('path')[idx])
        img = iio.imread(path)
        class_id = self.metadata_file.get_column('class_id')[idx]
        species = self.metadata_file.get_column('species')[idx].replace('_', ' ')
        # subcategory = self.metadata_file.get_column('subcategory')[idx]
        # if subcategory is not None:
        #     subcategory = subcategory.replace('_', ' ')
        photographer = self.metadata_file.get_column('photographer')[idx].replace('_', ' ')
        element = {
            'img': img,
            'class_id': class_id,
            'species': species,
            # 'subcategory': subcategory,
            'photographer': photographer,
        }
        return Dataset.from_dict(element)

nabirds_train = NABirdsDataset(metadata_train, cleaned_img_dir)
nabirds_val = NABirdsDataset(metadata_val, cleaned_img_dir)

class Normalize(grain.MapTransform):
    def map(self, element):
        img = element['img']
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        img = img.astype(np.float32) / 255.0
        img_norm = (img - mean) / std
        element['img'] = img_norm
        return element

class AsArray(grain.MapTransform):
    def map(self, element):
        element['class_id'] = np.asarray(element['class_id'])
        element['species'] = np.asarray(element['species'])
        # element['subcategory'] = np.asarray(element['subcategory'])
        element['photographer'] = np.asarray(element['photographer'])
        return element

seed = 123
train_batch_size = 32
val_batch_size = 2 * train_batch_size

train_sampler = grain.IndexSampler(
    num_records=len(nabirds_train),
    shuffle=True,                      # We shuffle the training set
    seed=seed,
    shard_options=grain.NoSharding(),  # No sharding for a single-device setup
    num_epochs=1
)

train_loader = grain.DataLoader(
    data_source=nabirds_train,
    sampler=train_sampler,
    operations=[
        Normalize(),
        AsArray(),
        grain.Batch(train_batch_size, drop_remainder=True)
    ]
)

train_batch = next(iter(train_loader))
