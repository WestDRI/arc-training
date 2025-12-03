import os
import polars as pl
import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import grain.python as grain
import jax.numpy as jnp
import dm_pix as pix
from jax import random

with open("4_augmentation.py") as file:
    exec(file.read())

sampler = grain.SequentialSampler(
    num_records=4,
    shard_options=grain.NoSharding()
)

for record_metadata in sampler:
    print(record_metadata)

data_loader = grain.DataLoader(
    data_source=nabirds_train,
    operations=transformations,
    sampler=sampler,
    worker_count=0
)

fig = plt.figure()

for i, element in enumerate(data_loader):
    ax = plt.subplot(2, 2, i + 1)
    plt.tight_layout()
    ax.set_title(
        f'element {i}, identification: {element['id']}, picture by {element['photographer']}'
    )
    ax.axis('off')
    plt.imshow(element['image'])

plt.show()

# PyTorch (verify)
# dataloader = DataLoader(nabirds_bb_cropped_train, batch_size=4,
#                         shuffle=False, num_workers=0)

def show_batch(sample_batched):
    """Show a batch of samples."""
    images_batch = sample_batched['image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')
