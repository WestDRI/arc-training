import os
import polars as pl

import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

with open("1_metadata.py") as file:
    exec(file.read())

class NABirdsDataset:
    """NABirds dataset class."""
    def __init__(self, metadata_file, data_dir):
        self.metadata = metadata_file
        self.data_dir = data_dir
    def __len__(self):
        return len(self.metadata)
    def __getitem__(self, idx):
        img_path = os.path.join(
            self.data_dir,
            self.metadata.get_column('path')[idx]
        )
        img = iio.imread(img_path)
        img_id = self.metadata.get_column('id')[idx].replace('_', ' ')
        img_photographer = self.metadata.get_column('photographer')[idx].replace('_', ' ')
        img_bb_x = self.metadata.get_column('bb_x')[idx]
        img_bb_y = self.metadata.get_column('bb_y')[idx]
        img_bb_width = self.metadata.get_column('bb_width')[idx]
        img_bb_height = self.metadata.get_column('bb_height')[idx]
        element = {
            'image': img,
            'id': img_id,
            'photographer': img_photographer,
            'bbx' : img_bb_x,
            'bby' : img_bb_y,
            'bbwidth' : img_bb_width,
            'bbheight' : img_bb_height
        }
        return element

nabirds_train = NABirdsDataset(
    metadata_train,
    img_dir
)

fig = plt.figure(figsize=(8, 8))

for i, element in enumerate(nabirds_train):
    ax = plt.subplot(2, 2, i + 1)
    plt.tight_layout()
    ax.set_title(
        f'Element {i}\nIdentification: {element['id']}\nPicture by {element['photographer']}',
        fontsize=9
    )
    ax.axis('off')
    plt.imshow(element['image'])
    rect = patches.Rectangle(
        (element['bbx'], element['bby']),
        element['bbwidth'],
        element['bbheight'],
        linewidth=1,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)
    if i == 3:
        plt.show()
        break

for i, element in enumerate(nabirds_train):
    print(f'Image dimensions: {element['image'].shape}, data type: {element['image'].dtype}'
    )
    if i == 3:
        break
