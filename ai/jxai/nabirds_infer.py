import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
import imageio.v3 as iio
import polars as pl
import dm_pix as pix

# Import from jxai
try:
    from jxai import VisionTransformer
except ImportError:
    # If jxai is not in pythonpath, try to append current dir
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from jxai import VisionTransformer

# Constants
CHECKPOINT_DIR = os.path.abspath('checkpoint')
METADATA_PATH = 'metadata.parquet'

def load_model(checkpoint_dir=CHECKPOINT):
    """Loads the model from the checkpoint directory."""
    print(f"Loading model from {checkpoint_dir}...")

    # Instantiate model with same structure as training (405 classes for NABirds)
    model = VisionTransformer(num_classes=405)

    # Create abstract state for structure
    abstract_state = nnx.state(model)

    # Create checkpointer
    checkpointer = ocp.PyTreeCheckpointer()

    # Define restore args
    restore_args = ocp.args.PyTreeRestore(abstract_state)

    # Restore
    try:
        restored_state = checkpointer.restore(checkpoint_dir, item=restore_args)
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

def load_species_mapping(metadata_path=METADATA_PATH):
    """Loads species ID to name mapping from metadata."""
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found at {metadata_path}")
        return {}

    df = pl.read_parquet(metadata_path)
    # Creates specific id -> name mapping
    mapping = dict(df.select(['species_id', 'species_name']).unique().iter_rows())
    return mapping

def preprocess_image(image_path):
    """Reads and preprocesses an image for the model."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = iio.imread(image_path)

    # Ensure 3 channels (RGB)
    if img.ndim == 2: # Grayscale
        img = img[..., None]
        img = np.repeat(img, 3, axis=-1)
    elif img.shape[-1] == 4: # RGBA
        img = img[..., :3]

    # Convert to jax array and normalize to [0, 1]
    img = jnp.array(img).astype(jnp.float32) / 255.0

    # Resize to 224x224
    img = jax.image.resize(img, (224, 224, 3), method='bilinear')

    # Normalize with mean/std (matching training logic: ZScore transform)
    # In jxai.py: mean = 0.5, std = 0.5
    mean = jnp.array([0.5, 0.5, 0.5])
    std = jnp.array([0.5, 0.5, 0.5])
    img = (img - mean) / std

    # Add batch dimension
    img = img[None, ...]

    return img

def predict(image_path):
    """Runs prediction on a single image."""
    model = load_model()
    mapping = load_species_mapping()

    print(f"Processing image: {image_path}")
    img = preprocess_image(image_path)

    # Inference
    logits = model(img)
    probs = nnx.softmax(logits)

    # Get top prediction
    predicted_id = int(jnp.argmax(probs))
    confidence = float(jnp.max(probs))

    predicted_name = mapping.get(predicted_id, f"Unknown ID {predicted_id}")

    print("-" * 30)
    print(f"Prediction: {predicted_name}")
    print(f"Species ID: {predicted_id}")
    print(f"Confidence: {confidence:.2%}")
    print("-" * 30)

    # Top 5
    top_k = 5
    top_indices = jnp.argsort(probs, descending=True)[0, :top_k]
    print(f"Top {top_k} predictions:")
    for idx in top_indices:
        idx = int(idx)
        score = float(probs[0, idx])
        name = mapping.get(idx, f"ID {idx}")
        print(f"  {name}: {score:.2%}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict(image_path)
    else:
        print("Usage: uv run python inference.py <path_to_bird_image>")
