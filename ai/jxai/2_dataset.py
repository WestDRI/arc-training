import imageio.v3 as iio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import grain.python as grain

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
        sample = {
            'image': img,
            'id': img_id,
            'photographer': img_photographer,
            'bbx' : img_bb_x,
            'bby' : img_bb_y,
            'bbwidth' : img_bb_width,
            'bbheight' : img_bb_height
        }
        return sample


nabirds_train = NABirdsDataset(
    metadata_train,
    os.path.join(base_dir, img_dir)
    )

# fig = plt.figure()

# for i, sample in enumerate(nabirds_train):
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title(
#         f'Sample {i}, identification: {sample['id']}, picture by {sample['photographer']}'
#     )
#     ax.axis('off')
#     plt.imshow(sample['image'])
#     rect = patches.Rectangle(
#         (sample['bbx'], sample['bby']),
#         sample['bbwidth'],
#         sample['bbheight'],
#         linewidth=2,
#         edgecolor='r',
#         facecolor='none'
#     )
#     ax.add_patch(rect)
#     if i == 3:
#         plt.show()
#         break

for i, sample in enumerate(nabirds_train):
    print(f'Image dimensions: {sample['image'].shape}, data type: {sample['image'].dtype}'
    )
    if i == 3:
        break
