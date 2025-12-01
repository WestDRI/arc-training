import os
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import imageio.v3 as iio
# from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import dm_pix as pix

base_dir = '/home/marie/parvus/ptmp/jxbirds/nabirds'
img_dir = os.path.join(base_dir, 'images')
# parts_dir = os.path.join(base_dir, 'parts')

# processed_img_dir = os.path.join(base_dir, 'processed_images')

# hierarchy_file = os.path.join(base_dir, 'hierarchy.txt')

bb_file = os.path.join(base_dir, 'bounding_boxes.txt')
classes_translation_file = os.path.join(base_dir, 'classes_fixed.txt')
class_labels_file = os.path.join(base_dir, 'image_class_labels.txt')
img_file = os.path.join(base_dir, 'images.txt')
photographers_file = os.path.join(base_dir, 'photographers_fixed.txt')
# sizes_file = os.path.join(base_dir, 'sizes.txt')
train_test_split_file = os.path.join(base_dir, 'train_test_split.txt')

bb = pl.read_csv(
    bb_file,
    separator=' ',
    has_header=False,
    new_columns=['UUID', 'bb_x', 'bb_y', 'bb_width', 'bb_height']
)

classes = pl.read_csv(
    class_labels_file,
    separator=' ',
    has_header=False,
    new_columns=['UUID', 'class']
)

classes_translation = pl.read_csv(
    classes_translation_file,
    separator=' ',
    has_header=False,
    new_columns=['class', 'id']
)

img_paths = pl.read_csv(
    img_file,
    separator=' ',
    has_header=False,
    new_columns=['UUID', 'path']
)

photographers = pl.read_csv(
    photographers_file,
    separator=' ',
    has_header=False,
    new_columns=['UUID', 'photographer']
)

# sizes = pl.read_csv(
#     sizes_file,
#     separator=' ',
#     has_header=False,
#     new_columns=['UUID', 'img_width', 'img_height']
# )

train_test_split = pl.read_csv(
    train_test_split_file,
    separator=' ',
    has_header=False,
    new_columns=['UUID', 'is_training_img']
)

classes_metadata = (
    classes.join(classes_translation, on='class')
)

metadata = (
    bb.join(classes_metadata, on='UUID')
    .join(img_paths, on='UUID')
    .join(photographers, on='UUID')
    # .join(sizes, on='UUID')
    .join(train_test_split, on='UUID')
)

metadata_train = metadata.filter(pl.col('is_training_img') == 1)

class NABirdsDataset(Dataset):
    """NABirds dataset class."""
    def __init__(self, metadata_file, data_dir, transform=None):
        self.metadata = metadata_file
        self.data_dir = data_dir
        self.transform = transform
    def __len__(self):
        return len(self.metadata)
    def __getitem__(self, idx):
        img_path = os.path.join(
            self.data_dir,
            self.metadata.get_column('path')[idx]
        )
        img = iio.imread(img_path)
        # img = Image.open(img_path)
        img_id = self.metadata.get_column('id')[idx].replace('_', ' ')
        img_photographer = self.metadata.get_column('photographer')[idx].replace('_', ' ')
        img_bb_x = self.metadata.get_column('bb_x')[idx]
        img_bb_y = self.metadata.get_column('bb_y')[idx]
        img_bb_width = self.metadata.get_column('bb_width')[idx]
        img_bb_height = self.metadata.get_column('bb_height')[idx]
        sample = {
            'image': img,
            'id': img_id,
            'photographer': img_photographer,
            'bb' : (img_bb_x, img_bb_y, img_bb_width, img_bb_height)
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

nabirds_train = NABirdsDataset(
    metadata_train,
    os.path.join(base_dir, img_dir)
    )

fig = plt.figure()

for i, sample in enumerate(nabirds_train):
    print(i, sample['image'].shape)
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title(
        f'Sample {i}, identification: {sample['id']}, picture by {sample['photographer']}'
    )
    ax.axis('off')
    plt.imshow(sample['image'])
    rect = patches.Rectangle(
        (sample['bb'][0], sample['bb'][1]),
        sample['bb'][2],
        sample['bb'][3],
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)
    if i == 3:
        plt.show()
        break



# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#     def __call__(self, sample):
#         img = sample['image'],
#         img_id = sample['id'],
#         img_photographer = sample['photographer'],
#         img_bb_x = sample['bb'][0],
#         img_bb_y = sample['bb'][1],
#         img_bb_width = sample['bb'][2],
#         img_bb_height = sample['bb'][3],
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C x H x W
#         img = img.transpose((2, 0, 1))
#         sample = {
#             'image': torch.from_numpy(img),
#             'id': img_id,
#             'photographer': img_photographer,
#             'bb': (img_bb_x, img_bb_y, img_bb_width, img_bb_height),
#         }
#         return sample


class BBCrop(object):
    def __call__(self, sample):
        img = sample['image'],
        img_id = sample['id'],
        img_photographer = sample['photographer'],
        img_bb_x = sample['bb'][0],
        img_bb_y = sample['bb'][1],
        img_bb_width = sample['bb'][2],
        img_bb_height = sample['bb'][3],
        img = img[img_bb_y:img_bb_y+img_bb_height, img_bb_x:img_bb_x+img_bb_width]
        sample = {
            'image': img,
            'id': img_id,
            'photographer': img_photographer
        }
        return sample


data_dir = os.path.join(base_dir, img_dir)
img_path = os.path.join(data_dir, metadata.get_column("path")[0])
img = iio.imread(img_path)
# img = Image.open(img_path)
img_id = metadata.get_column("id")[0].replace("_", " ")
img_photographer = metadata.get_column("photographer")[0].replace("_", " ")
img_bb_x = metadata.get_column("bb_x")[0]
img_bb_y = metadata.get_column("bb_y")[0]
img_bb_width = metadata.get_column("bb_width")[0]
img_bb_height = metadata.get_column("bb_height")[0]
sample = {
    "image": img,
    "id": img_id,
    "photographer": img_photographer,
    "bb": (img_bb_x, img_bb_y, img_bb_width, img_bb_height),
}

img_cropped = img[img_bb_y:img_bb_y+img_bb_height, img_bb_x:img_bb_x+img_bb_width]






nabirds_bb_cropped_train = NABirdsDataset(
    metadata_train,
    os.path.join(base_dir, img_dir),
    transform=BBCrop()
)


# nabirds_bb_cropped_train = NABirdsDataset(
#     metadata_train,
#     os.path.join(base_dir, img_dir),
#     transforms.functional.crop(
#         sample["image"],
#         sample["bb"][0],
#         sample["bb"][1],
#         sample["bb"][2],
#         sample["bb"][3],
#     ),
# )


dataloader = DataLoader(nabirds_bb_cropped_train, batch_size=4,
                        shuffle=False, num_workers=0)



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






for i, sample in enumerate(nabirds_bb_cropped_train):
    print(i, sample['image'].shape)
    if i == 3:
        break



fig = plt.figure()

for i, sample in enumerate(nabirds_bb_cropped_train):
    print(i, sample['image'].shape)
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title(
        f'Sample {i}, identification: {sample['id']}, picture by {sample['photographer']}'
    )
    ax.axis('off')
    plt.imshow(sample['image'])
    if i == 3:
        plt.show()
        break




class NumpyLoader(DataLoader):
    """Custom DataLoader to return NumPy arrays from a PyTorch Dataset."""
    def __init__(self, dataset, batch_size=1,
                  shuffle=False, sampler=None,
                  batch_sampler=None, num_workers=0,
                  pin_memory=False, drop_last=False,
                  timeout=0, worker_init_fn=None):
      super(self.__class__, self).__init__(dataset,
          batch_size=batch_size,
          shuffle=shuffle,
          sampler=sampler,
          batch_sampler=batch_sampler,
          num_workers=num_workers,
          pin_memory=pin_memory,
          drop_last=drop_last,
          timeout=timeout,
          worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
  """Transform class to flatten and cast images to float32."""
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))






from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# These random transforms are re-calculated every time an image is fetched
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224), # Different crop every epoch
    transforms.RandomHorizontalFlip(), # Different flip every epoch
    transforms.ToTensor(),
])

dataset = datasets.FakeData(transform=train_transforms) # Replaces your actual dataset
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# No extra code needed here; PyTorch does it automatically
for epoch in range(5):
    for batch in loader:
        # 'batch' contains newly augmented versions of the data
        pass


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class EpochAwareDataset(Dataset):
    def __init__(self, data, transform_dict):
        """
        data: Your actual data list
        transform_dict: A dictionary mapping epoch numbers to transform pipelines
        """
        self.data = data
        self.transform_dict = transform_dict
        self.current_epoch = 0
        self.default_transform = transforms.ToTensor()

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Create a dummy image for demonstration
        # In reality, you would load self.data[idx]
        img = Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))

        # Select transform based on current epoch
        # Fallback to default if epoch not in dict
        active_transform = self.transform_dict.get(self.current_epoch, self.default_transform)
        
        return active_transform(img)

# 1. Define different transforms for different phases
transforms_epoch_0 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

transforms_epoch_1 = transforms.Compose([
    transforms.RandomRotation(degrees=90), # Only rotate in epoch 1
    transforms.ToTensor()
])

transforms_epoch_2 = transforms.Compose([
    transforms.ColorJitter(brightness=1), # Only jitter in epoch 2
    transforms.ToTensor()
])

# Map epochs to transforms
transform_schedule = {
    0: transforms_epoch_0,
    1: transforms_epoch_1,
    2: transforms_epoch_2
}

# 2. Initialize
# Dummy data list
data = list(range(10)) 
dataset = EpochAwareDataset(data, transform_schedule)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 3. Training Loop
for epoch in range(3):
    # CRITICAL STEP: Update the epoch in the dataset
    loader.dataset.set_epoch(epoch)
    
    print(f"\n--- Epoch {epoch} ---")
    print(f"Active Transform: {loader.dataset.transform_dict.get(epoch)}")
    
    for batch_idx, images in enumerate(loader):
        # Your training code here
        pass

# More robust method in case I use several workers

class EpochAwareDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = None # Will be set externally

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# In your loop:
for epoch in range(epochs):
    # Re-assign the transform attribute directly before iteration
    dataset.transform = get_transform_for_epoch(epoch)
    
    for batch in loader:
        ...

