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

nabirds_train_seqsampler = grain.SequentialSampler(
    num_records=4,
    shard_options=grain.NoSharding()
)

for record_metadata in nabirds_train_seqsampler:
    print(record_metadata)

nabirds_train_dl = grain.DataLoader(
    data_source=nabirds_train,
    operations=transformations,
    sampler=nabirds_train_seqsampler,
    worker_count=0
)

fig = plt.figure(figsize=(8, 8))

for i, element in enumerate(nabirds_train_dl):
    ax = plt.subplot(2, 2, i + 1)
    plt.tight_layout()
    ax.set_title(
        f'Element {i}\nIdentification: {element['id']}\nPicture by {element['photographer']}',
        fontsize=9
    )
    ax.axis('off')
    plt.imshow(element['image'])

plt.show()

# PyTorch (verify)
# dataloader = DataLoader(nabirds_bb_cropped_train, batch_size=4,
#                         shuffle=False, num_workers=0)

def show_batch(elements_batched):
    """Show a batch of elements."""
    images_batch = elements_batched['image']
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
