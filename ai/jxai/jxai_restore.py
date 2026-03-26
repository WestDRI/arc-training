import os
import sys
import polars as pl
import imageio.v3 as iio
import grain.python as grain
import jax
from jax import random
import dm_pix as pix
import numpy as np
import jax.numpy as jnp
from flax import nnx
import matplotlib.pyplot as plt
import orbax.checkpoint as ocp
import pickle

# base_dir = 'nabirds'
# cleaned_img_dir = os.path.join(base_dir, 'cleaned_images')

# metadata = pl.read_parquet('metadata.parquet')
# metadata_train = metadata.filter(pl.col('is_training_img') == 1)
# metadata_val = metadata.filter(pl.col('is_training_img') == 0)

# class NABirdsDataset:
#     """NABirds dataset class."""
#     def __init__(self, metadata, data_dir):
#         self.metadata = metadata
#         self.data_dir = data_dir
#     def __len__(self):
#         return len(self.metadata)
#     def __getitem__(self, idx):
#         path = os.path.join(self.data_dir, self.metadata.get_column('path')[idx])
#         img = iio.imread(path)
#         species_name = self.metadata.get_column('species_name')[idx]
#         species_id = self.metadata.get_column('species_id')[idx]
#         photographer = self.metadata.get_column('photographer')[idx]
#         return {
#             'img': img,
#             'species_name': species_name,
#             'species_id': species_id,
#             'photographer': photographer,
#         }

# nabirds_train = NABirdsDataset(metadata_train, cleaned_img_dir)
# nabirds_val = NABirdsDataset(metadata_val, cleaned_img_dir)

# class Normalize(grain.MapTransform):
#     def map(self, element):
#         img = element['img']
#         mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
#         std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
#         img = img.astype(np.float32) / 255.0
#         img_norm = (img - mean) / std
#         element['img'] = img_norm
#         return element

# class ToFloat(grain.MapTransform):
#     def map(self, element):
#         element['img'] = element['img'].astype(np.float32) / 255.0
#         return element

# key = random.key(31)
# key, subkey1, subkey2, subkey3, subkey4 = random.split(key, num=5)

# class RandomCrop(grain.MapTransform):
#     def map(self, element):
#         element['img'] = pix.random_crop(
#             key=subkey1,
#             image=element['img'],
#             crop_sizes=(224, 224, 3)
#         )
#         return element

# class RandomFlip(grain.MapTransform):
#     def map(self, element):
#         element['img'] = pix.random_flip_left_right(
#             key=subkey2,
#             image=element['img']
#         )
#         return element

# class RandomContrast(grain.MapTransform):
#     def map(self, element):
#         element['img'] = pix.random_contrast(
#             key=subkey3,
#             image=element['img'],
#             lower=0.8,
#             upper=1.2
#         )
#         return element

# class RandomGamma(grain.MapTransform):
#     def map(self, element):
#         element['img'] = pix.random_gamma(
#             key=subkey4,
#             image=element['img'],
#             min_gamma=0.6,
#             max_gamma=1.2
#         )
#         return element

# class ZScore(grain.MapTransform):
#     def map(self, element):
#         img = element['img']
#         mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
#         std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
#         img = (img - mean) / std
#         element['img'] = img
#         return element

# seed = 123
# train_batch_size = 8
# val_batch_size = 2 * train_batch_size

# train_sampler = grain.IndexSampler(
#     num_records=len(nabirds_train),
#     shuffle=True,
#     seed=seed,
#     shard_options=grain.NoSharding(),
#     num_epochs=1
# )

# train_loader = grain.DataLoader(
#     data_source=nabirds_train,
#     sampler=train_sampler,
#     operations=[
#         ToFloat(),
#         RandomCrop(),
#         RandomFlip(),
#         RandomContrast(),
#         RandomGamma(),
#         ZScore(),
#         grain.Batch(train_batch_size, drop_remainder=True)
#     ]
# )

# val_sampler = grain.IndexSampler(
#     num_records=len(nabirds_val),
#     shuffle=False,
#     seed=seed,
#     shard_options=grain.NoSharding(),
#     num_epochs=1
# )

# val_loader = grain.DataLoader(
#     data_source=nabirds_val,
#     sampler=val_sampler,
#     operations=[
#         Normalize(),
#         grain.Batch(val_batch_size)
#     ]
# )

# with open('train_metrics.pkl', 'rb') as f:
#     train_metrics_history = pickle.load(f)

# with open('eval_metrics.pkl', 'rb') as f:
#     eval_metrics_history = pickle.load(f)

# plt.plot(train_metrics_history['train_loss'], label='Loss value during training')
# plt.legend()

# plt.savefig('training_loss.png', dpi=400)

# fig, axs = plt.subplots(1, 2, figsize=(10, 10))
# axs[0].set_title('Loss value on validation set')
# axs[0].plot(eval_metrics_history['val_loss'])
# axs[1].set_title('Accuracy on validation set')
# axs[1].plot(eval_metrics_history['val_accuracy'])

# Import from nabirds_train
try:
    from nabirds_train import VisionTransformer
except ImportError:
    # If nabirds_train is not in pythonpath, try to append current dir
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from nabirds_train import VisionTransformer

checkpoint_dir = '/home/marie/parvus/prog/mint/ai/jxai/checkpoints/'

def load_model(checkpoint_dir=checkpoint_dir):
    """Loads the model from the checkpoint directory."""
    print(f"Loading model from {checkpoint_dir}...")
    # Instantiate model with same structure as training (405 classes for NABirds)
    model = VisionTransformer(num_classes=405)
    # Create abstract state for structure
    state = nnx.state(model)
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
    # Create checkpointer
    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    mngr = ocp.CheckpointManager(checkpoint_dir, options=options, item_names=('state', 'meta'))
    # Define restore args
    # restore_args = ocp.args.PyTreeRestore(abstract_state)
    # Restore
    try:
        restored_state = mngr.restore(mngr.latest_step(), args=ocp.args.Composite(state=ocp.args.StandardRestore(abstract_state)))
        # Update model with restored state
        nnx.update(model, restored_state)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have trained the model and a checkpoint exists.")
        sys.exit(1)
    # Set to eval mode
    model.eval()
    return model

# def load_model(checkpoint_dir=checkpoint_dir):
#     """Loads the model from the checkpoint directory."""
#     print(f"Loading model from {checkpoint_dir}...")
#     # Instantiate model with same structure as training (405 classes for NABirds)
#     model = VisionTransformer(num_classes=405)
#     # Create abstract state for structure
#     state = nnx.state(model)
#     abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
#     # Create checkpointer
#     options = ocp.CheckpointManagerOptions(max_to_keep=3)
#     mngr = ocp.CheckpointManager(checkpoint_dir, options=options)
#     # Define restore args
#     # restore_args = ocp.args.PyTreeRestore(abstract_state)
#     # Restore
#     try:
#         restored_state = mngr.restore(mngr.latest_step(), abstract_state)
#         # Update model with restored state
#         nnx.update(model, restored_state)
#         print("Model loaded successfully.")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         print("Ensure you have trained the model and a checkpoint exists.")
#         sys.exit(1)
#     # Set to eval mode
#     model.eval()
#     return model

# def load_model(checkpoint_dir=checkpoint_dir):
#     """Loads the model from the checkpoint directory."""
#     print(f"Loading model from {checkpoint_dir}...")
#     # Instantiate model with same structure as training (405 classes for NABirds)
#     model = VisionTransformer(num_classes=405)
#     # Create abstract state for structure
#     state = nnx.state(model)
#     abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
#     # Create checkpointer
#     options = ocp.CheckpointManagerOptions(max_to_keep=3)
#     mngr = ocp.CheckpointManager(checkpoint_dir, options=options)
#     # Define restore args
#     # restore_args = ocp.args.PyTreeRestore(abstract_state)
#     # Restore
#     try:
#         restored_state = mngr.restore(mngr.latest_step(), abstract_state)
#         # Update model with restored state
#         nnx.update(model, restored_state)
#         print("Model loaded successfully.")
#     except Exception as e:
#         print(f"{e}")
#         # sys.exit(1)
#     # Set to eval mode
#     model.eval()
#     return model

# def load_model(checkpoint_dir=checkpoint_dir):
#     """Loads the model from the checkpoint directory."""
#     print(f"Loading model from {checkpoint_dir}...")
#     # Instantiate model with same structure as training (405 classes for NABirds)
#     model = VisionTransformer(num_classes=405)
#     # Create abstract state for structure
#     state = nnx.state(model)
#     # 3. Create an abstract instance of the model
#     #    This creates the shape/structure without allocating heavy arrays.
#     abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
#     # Create checkpointer
#     options = ocp.CheckpointManagerOptions(max_to_keep=3)
#     mngr = ocp.CheckpointManager(checkpoint_dir, options=options)
#     # 4. Split the abstract model to get the 'abstract state'
#     #    'graphdef' holds the static structure, 'abstract_state' holds the     shapes/types.
#     graphdef, abstract_state = nnx.split(abstract_state)
#     # 5. Restore the checkpoint
#     #    We tell Orbax to look for the latest step and use 'abstract_state' as the schema.
#     latest_step = mngr.latest_step()
#     if latest_step is not None:
#         restored_state = mngr.restore(
#             latest_step,
#             args=ocp.args.StandardRestore(abstract_state)
#         )
#         # 6. Merge the restored state back into a real model
#         model = nnx.merge(graphdef, restored_state)
#         print("Model restored successfully!")
#     else:
#         print("No checkpoint found.")


# def load_model(checkpoint_dir=checkpoint_dir):
#     """Loads the model from the checkpoint directory."""
#     print(f"Loading model from {checkpoint_dir}...")
#     # Instantiate model with same structure as training (405 classes for NABirds)
#     model = VisionTransformer(num_classes=405)
#     # Create abstract state for structure
#     state = nnx.state(model)
#     # 3. Create an abstract instance of the model
#     #    This creates the shape/structure without allocating heavy arrays.
#     abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
#     # Create checkpointer
#     options = ocp.CheckpointManagerOptions(max_to_keep=3)
#     mngr = ocp.CheckpointManager(checkpoint_dir, options=options)
#     # 4. Split the abstract model to get the 'abstract state'
#     #    'graphdef' holds the static structure, 'abstract_state' holds the     shapes/types.
#     graphdef, abstract_state = nnx.split(abstract_state)
#     # 5. Restore the checkpoint
#     #    We tell Orbax to look for the latest step and use 'abstract_state' as the schema.
#     latest_step = mngr.latest_step()
#     if latest_step is not None:
#         restored_state = mngr.restore(
#             latest_step,
#             args=ocp.args.StandardRestore(abstract_state)
#         )
#         # 6. Merge the restored state back into a real model
#         model = nnx.merge(graphdef, restored_state)
#         print("Model restored successfully!")
#     else:
#         print("No checkpoint found.")

model = load_model()
# test_indices = [250, 500, 750, 1000]
# test_images = jnp.array([nabirds_val[i]['img'] for i in test_indices])
# expected_labels = [nabirds_val[i]['species_name'] for i in test_indices]

# preds = model(test_images)
# probas = nnx.softmax(preds, axis=1)
# pred_labels = probas.argmax(axis=1)

# def translator(df, species_id):
#     species_name = df.unique(subset='species_id').filter(
#         pl.col('species_id') == species_id
#     ).select(pl.col('species_name')).item()
#     return species_name

# num_samples = len(test_indices)

# fig, axs = plt.subplots(1, num_samples, figsize=(7, 2))

# for i in range(num_samples):
#     img, expected_label = test_images[i], expected_labels[i]
#     pred_label_id = pred_labels[i].item()
#     pred_label_name = translator(metadata, pred_label_id)
#     proba = probas[i, pred_label_id].item()
#     if img.dtype in (np.float32, ):
#         img = ((img - img.min()) / (img.max() - img.min()) * 255.0).astype(np.uint8)
#     plt.tight_layout()
#     axs[i].set_title(
#         f"""
#         Expected: {expected_labels[i]}
#         Predicted: {pred_label_name}
#         p={proba:.2f}
#         """,
#         fontsize=6.5,
#         linespacing=1.5
#     )
#     axs[i].axis('off')
#     axs[i].imshow(img)

# plt.savefig('sample_tests.png', dpi=300)
