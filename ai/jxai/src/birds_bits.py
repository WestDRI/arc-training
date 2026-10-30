import os
import polars as pl
import shutil
import cv2
import dm_pix as pix
from PIL import Image
import numpy as np

# Define the path to your image file
image_path = 'your_image.png'  # Replace with the actual path to your image

# Open the image file using Pillow
try:
    img = Image.open(image_path)
except FileNotFoundError:
    print(f"Error: The image file '{image_path}' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while opening the image: {e}")
    exit()

# Convert the PIL Image object to a NumPy array
image_array = np.array(img)

# Print the shape of the array to verify (optional)
print(f"Shape of the image array: {image_array.shape}")

# You can now work with 'image_array' as a standard NumPy array
# For example, to display the image (requires matplotlib):
# import matplotlib.pyplot as plt
# plt.imshow(image_array)
# plt.show()

base_dir = "/home/marie/parvus/ptmp/jxbirds/nabirds"
img_dir = os.path.join(base_dir, "images")
parts_dir = os.path.join(base_dir, "parts")

processed_img_dir = os.path.join(base_dir, "processed_images")

# classes_file = os.path.join(base_dir, "classes.txt")
# hierarchy_file = os.path.join(base_dir, "hierarchy.txt")

bb_file = os.path.join(base_dir, "bounding_boxes.txt")
class_labels_file = os.path.join(base_dir, "image_class_labels.txt")
img_file = os.path.join(base_dir, "images.txt")
photographers_file = os.path.join(base_dir, "photographers.txt")
sizes_file = os.path.join(base_dir, "sizes.txt")
train_test_split_file = os.path.join(base_dir, "train_test_split.txt")

# bb = pl.read_csv(
#     bb_file,
#     separator=" ",
#     has_header=False,
#     new_columns=["UUID", "bb_x", "bb_y", "bb_width", "bb_height"],
#     schema={
#         "UUID": pl.Array(pl.String, 48_562),
#         "bb_x": pl.Array(pl.Int64, 48_562),
#         "bb_y": pl.Array(pl.Int64, 48_562),
#         "bb_width": pl.Array(pl.Int64, 48_562),
#         "bb_height": pl.Array(pl.Int64, 48_562),
#         }
# )

bb = pl.read_csv(
    bb_file,
    separator=" ",
    has_header=False,
    new_columns=["UUID", "bb_x", "bb_y", "bb_width", "bb_height"]
)

classes = pl.read_csv(
    class_labels_file,
    separator=" ",
    has_header=False,
    new_columns=["UUID", "class"],
)

img_paths = pl.read_csv(
    img_file,
    separator=" ",
    has_header=False,
    new_columns=["UUID", "path"]
)

# photographers = pl.read_csv(
#     photographers_file,
#     separator=" ",
#     has_header=False,
#     new_columns=["UUID", "photographer"]
# )

sizes = pl.read_csv(
    sizes_file,
    separator=" ",
    has_header=False,
    new_columns=["UUID", "img_width", "img_height"]
)

train_test_split = pl.read_csv(
    train_test_split_file,
    separator=" ",
    has_header=False,
    new_columns=["UUID", "is_training_img"]
)

metadata = (
    bb.join(classes, on="UUID")
    .join(img_paths, on="UUID")
    .join(sizes, on="UUID")
    .join(train_test_split, on="UUID")
)

# Step 1: Find the largest bounding box dimensions
max_bb_width = metadata.select(pl.col("bb_width")).max().item()
max_bb_height = metadata.select(pl.col("bb_height")).max().item()
largest_box_size = max(max_bb_width, max_bb_height)

print(f"Largest box dimensions: Width={max_bb_width}, Height={max_bb_height}, Side={largest_box_size}")

os.makedirs(processed_img_dir, exist_ok=True)

# or:

if os.path.exists(processed_img_dir):
    shutil.rmtree(processed_img_dir)  # Clear processed images directory
os.makedirs(processed_img_dir, exist_ok=True)

def crop_and_save_img(img_path, bb_x, bb_y, bb_width, bb_height, save_path, box_size):
    """Crop the image around the bird's bounding box and resize to uniform dimensions."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        return False

    img_height, img_width, _ = img.shape

    # Calculate the bird's center
    bird_center_x = bb_x + bb_width / 2
    bird_center_y = bb_y + bb_height / 2

    ### New stuff from Kevin O'Neill
    max_side = max(bb_width, bb_height)

    # Calculate the box coordinates (centered on bird and fit within image bounds)
    x_min = max(0, int(bird_center_x - max_side / 2))
    y_min = max(0, int(bird_center_y - max_side / 2))
    x_max = min(img_width, int(bird_center_x + max_side / 2))
    y_max = min(img_height, int(bird_center_y + max_side / 2))

    # Crop and resize the image
    cropped_img = img[y_min:y_max, x_min:x_max]
    resized_img = cv2.resize(cropped_img, (box_size, box_size))

    # Save the processed image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, resized_img)
    return True

# Test function on one image
crop_and_save_img(
    "/home/marie/parvus/ptmp/jxbirds/nabirds/images/0817/0000139e21dc4d0cbfe14cae3c85c829.jpg",
    83,
    59,
    128,
    228,
    "/home/marie/parvus/ptmp/jxbirds/nabirds/processed_images/0817/0000139e21dc4d0cbfe14cae3c85c829.jpg",
    224,
)

# map(
#     crop_and_save_img(
#         metadata_increased.get_column("img_path"),
#         metadata_increased.get_column("bb_x"),
#         metadata_increased.get_column("bb_y"),
#         metadata_increased.get_column("bb_width"),
#         metadata_increased.get_column("bb_height"),
#         metadata_increased.get_column("save_path"),
#         224
#         )
#     )

metadata_increased = metadata.with_columns(
    img_path=pl.lit(img_dir) + pl.lit("/") + pl.col("path"),
    save_path=pl.lit(processed_img_dir) + pl.lit("/") + pl.col("path"),
)

# Process all images
for row in metadata_increased.rows(named=True):
    crop_and_save_img(
        row["img_path"],
        row["bb_x"],
        row["bb_y"],
        row["bb_width"],
        row["bb_height"],
        row["save_path"],
        224,
    )

print("All images cropped and resized to uniform dimensions.")


# load images with Pillow
for path in train_paths_test:
    full_path = os.path.join(base_dir, img_dir, path)
    # Open the image file using Pillow
    try:
        img = Image.open(full_path)
    except FileNotFoundError:
        print(f"Error: The image file '{full_path}' was not found.")
        exit()
    except Exception as e:
        print(f"An error occurred while opening the image: {e}")
        exit()
    # Convert the PIL Image object to a NumPy array
    img_array = np.array(img, dtype=np.float32)
    train_imgs_test.append(img_array)


# Dataloader with Grain and Pillow:
class Dataset:
    def __init__(self, data_dir, paths):
        self.data_dir = data_dir
        self.paths = paths
        self.load_data()
    def load_data(self):
        self.dataset = []
        for path in self.paths:
            full_path = os.path.join(self.data_dir, path)
            # Open the image file using Pillow
            try:
                img = Image.open(full_path)
            except FileNotFoundError:
                print(f"Error: The image file '{full_path}' was not found.")
                exit()
            except Exception as e:
                print(f"An error occurred while opening the image: {e}")
                exit()
            # Convert the PIL Image object to a NumPy array
            img_array = np.array(img, dtype=np.float32)
            self.dataset.append(img_array)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        img = self.dataset[index]
        return np.ravel(np.array(img, dtype=np.float32))























transfer_model = tf.keras.applications.VGG16
transfer_model_shp = (224, 224)  # Typical for VGG16
transfer_model_input_shape = (224, 224, 3)
MyEpochs = 300
batch_size_val = 5
batch_size_data_load_val = 10
learning_rate_val = 0.001
dropout_rate = 0.25


# Filter only the classes that will be used in the model training
important_classes = {'295', '296', '297', '298', '299', '313', '314', '315', '316', '317'}
filtered_data = merged_data[merged_data['class'].isin(important_classes)].reset_index(drop=True)

filtered_data['path'] = filtered_data['path'].apply(lambda x: os.path.join(my_birds_image_dir, x))
# Assuming `filtered_data` is your DataFrame
# Splitting into training and testing based on `is_training_image`
train_data = filtered_data[filtered_data['is_training_image'] == 1]
test_data = filtered_data[filtered_data['is_training_image'] == 0]

# Define directories for organized data
train_dir = os.path.join(my_birds_image_dir, 'train')
test_dir = os.path.join(my_birds_image_dir, 'test')

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to copy images to target directory
def copy_images(df, target_dir):
    for _, row in df.iterrows():
        class_label = row['class']
        src_path = os.path.join(data_path, row['path'])  # Ensure the full path is resolved
        class_dir = os.path.join(target_dir, str(class_label))
        os.makedirs(class_dir, exist_ok=True)

        dest_path = os.path.join(class_dir, os.path.basename(src_path))
        if not os.path.exists(src_path):
            print(f"Warning: Source file not found: {src_path}")
            continue  # Skip this file if it doesn't exist

        shutil.copy(src_path, dest_path)

# Copy training and testing images
copy_images(train_data, train_dir)
copy_images(test_data, test_dir)

data_gen = ImageDataGenerator(
    rescale=1. / 255,                  # Normalize pixel values to [0, 1]
    rotation_range=30,                 # Randomly rotate images by up to 30 degrees
    width_shift_range=0.2,             # Randomly shift images horizontally
    height_shift_range=0.2,            # Randomly shift images vertically
    shear_range=0.2,                   # Apply shearing transformations
    zoom_range=0.2,                    # Randomly zoom in/out on images
    horizontal_flip=True,              # Randomly flip images horizontally
    vertical_flip=False,               # Optionally flip images vertically (unlikely for birds)
    brightness_range=[0.8, 1.2],       # Randomly adjust brightness
    fill_mode='nearest'                # Handle pixels outside the boundary of the image
)

# Training generators
train_data_shuffled_generator = data_gen.flow_from_directory(
    train_dir,
    target_size=transfer_model_shp,
    batch_size=batch_size_data_load_val,
    class_mode='categorical',
    shuffle=True
)

train_data_no_shuffle_generator = data_gen.flow_from_directory(
    train_dir,
    target_size=transfer_model_shp,
    batch_size=batch_size_data_load_val,
    class_mode='categorical',
    shuffle=False
)

# Test generator (using test_dir as a standalone validation directory)
test_data_generator = data_gen.flow_from_directory(
    test_dir,
    target_size=transfer_model_shp,
    batch_size=batch_size_data_load_val,
    shuffle=False,
    class_mode='categorical'
)

# Check the number of classes
numClasses = len(train_data_shuffled_generator.class_indices)  # Use class indices from train generator
print(f"Number of classes: {numClasses}")

cnn = transfer_model(
    include_top=False,
    weights="imagenet",
    input_shape=transfer_model_input_shape,
)
cnn.trainable = False ## using FREEZing loop
cnn.summary()

# global_avg_pooling_layer = GlobalAveragePooling2D()
flatten_layer = tf.keras.layers.Flatten()
dense_layer_1 = Dense(500, activation='relu', kernel_regularizer=l2(0.01))
batch_norm_1 = BatchNormalization()  # Batch normalization after the first dense layer
dropout_layer_1 = Dropout(dropout_rate)  # Add dropout after the first dense layer
dense_layer_2 = Dense(250, activation='relu', kernel_regularizer=l2(0.01))
batch_norm_2 = BatchNormalization()  # Batch normalization after the second dense layer
dropout_layer_2 = Dropout(dropout_rate)  # Add dropout after the second dense layer
dense_layer_3 = Dense(30, activation='relu', kernel_regularizer=l2(0.01))
dropout_layer_3 = Dropout(dropout_rate)  # Add dropout after the third dense layer
prediction_layer = Dense(numClasses, activation='softmax')  # Final output layer

Classifier = Sequential([
    cnn,                        # Convolutional base (e.g., VGG16, pre-trained or custom)
    flatten_layer,
    dense_layer_1,              # First dense layer
    batch_norm_1,               # Batch normalization for stability
    dropout_layer_1,            # Dropout after first dense layer
    dense_layer_2,              # Second dense layer
    batch_norm_2,               # Batch normalization for stability
    dropout_layer_2,            # Dropout after second dense layer
    dense_layer_3,              # Third dense layer
    dropout_layer_3,            # Dropout after third dense layer
    prediction_layer            # Output layer
])

Classifier.summary()

class EpochMetricsLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # Extract and log metrics
        metrics = {key: logs.get(key, 'nan') for key in ['accuracy', 'loss', 'val_accuracy', 'val_loss']}

        # Safeguard for NaN detection in loss
        val_loss = metrics['val_loss']
        if isinstance(val_loss, float) and math.isnan(val_loss):
            logger.warning(f"NaN detected in validation loss at epoch {epoch + 1}. Stopping training.")
            self.model.stop_training = True
            return

        # Log metrics
        logger.info(
            f"Epoch {epoch + 1}: "
            f"Train Accuracy: {metrics['accuracy']:.4f}, "
            f"Train Loss: {metrics['loss']:.4f}, "
            f"Validation Accuracy: {metrics['val_accuracy']:.4f}, "
            f"Validation Loss: {metrics['val_loss']:.4f}"
        )

# Define callbacks
callbacks = [
    ModelCheckpoint(
        filepath='best_vgg16_model.weights.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=50,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-4,
        verbose=1
    ),
    EpochMetricsLogger()
]

opt = tf.keras.optimizers.SGD(learning_rate=learning_rate_val)
Classifier.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=opt,
    metrics=['accuracy']
)

history = Classifier.fit(
    train_data_shuffled_generator,         # Training generator
    epochs=MyEpochs,                       # Number of epochs
    validation_data=test_data_generator,   # Validation generator
    shuffle=True,                          # Shuffling handled by the training generator
    verbose=0,                             # Verbose mode for training updates
    batch_size = batch_size_val,
    callbacks=callbacks,
)
