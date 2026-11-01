"""Draft of the code for the datasets page (pt2_datasets.qmd).

EuroSAT, the working dataset of the whole course: 27,000 tiny (64x64)
Sentinel-2 satellite images in 10 land-use classes. Three ways to hold
data for PyTorch, in the order of the page: a ready-made torchvision
dataset, the Hugging Face datasets library, and a custom Dataset for
our own files in folders.

The first run downloads the archive into ./data (once, ~90 MiB).
Credit: Helber et al. 2019 (https://github.com/phelber/eurosat, MIT
licence). Everything runs on CPU.

Run with: uv run python src/dataset.py
"""

from collections import Counter
from pathlib import Path

import datasets as hf
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision.datasets import EuroSAT


# ------------------------------------ 1. a ready-made torchvision dataset
# torchvision ships Dataset classes for many classic benchmarks. With
# download=True, the archive is fetched and extracted into ./data once;
# later runs find the files and skip the download.

eurosat = EuroSAT(root="data", download=True)

print(len(eurosat), "images")
print(eurosat.classes)

# EuroSAT is an ImageFolder: indexing returns one (image, label) pair
# and decodes just that file.
img, label = eurosat[0]
print(type(img).__name__, img.size, img.mode, "->", label, eurosat.classes[label])

# never trust a dataset to be balanced: count the labels
# (EuroSAT has 2,000 to 3,000 images per class: mild imbalance)
counts = Counter(eurosat.targets)
for i, name in enumerate(eurosat.classes):
    print(f"{name:>22} {counts[i]}")

# On disk, the layout is the classic "one folder per class", the most
# common way research image data arrives:
#   data/eurosat/2750/AnnualCrop/AnnualCrop_1.jpg
#   data/eurosat/2750/Forest/Forest_1.jpg
print(eurosat.samples[0])  # the (path, label) pairs ImageFolder scanned


# -------------------------------------------------------- 2. train/val split
# Never evaluate a model on data it trained on: split first. random_split
# returns two Subsets: thin views holding index lists into eurosat (no
# image is copied). Fractions are rounded to whole lengths.

train, val = random_split(
    eurosat, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
)
print(len(train), len(val), type(train).__name__)

# Subsets are Datasets in their own right: the next page wraps them in
# DataLoaders. random_split ignores class balance: with EuroSAT's mild
# imbalance (section 1) this is fine, but for small skewed data, draw a
# stratified split (same proportions per class in both parts) yourself.

# every later page must reuse this exact split, or validation numbers
# from different runs would not be comparable: save the indices once
torch.save({"train": train.indices, "val": val.indices}, "data/split.pt")


# -------------------------------------- 3. the same data through the HF hub
# The Hugging Face Hub mirrors many datasets and hosts research data
# directly. The datasets library adds caching under ~/.cache/huggingface,
# fast parallel preprocessing with .map, and streaming for data too big
# to store locally.

eurosat_hf = hf.load_dataset("blanchon/EuroSAT_RGB")  # DatasetDict of named splits
train_hf = eurosat_hf["train"]
print(train_hf)

# metadata comes with the data: labels are documented, not guessed
label_meta = train_hf.features["label"]  # a ClassLabel feature
print(label_meta.names)

# indexing returns a dict (not a tuple): column names instead of positions
print(train_hf[0]["label"], train_hf[0]["filename"])


# .map applies a function to every example, in parallel (num_proc=),
# and caches the result on disk: the second call returns instantly
def add_label_name(example):
    return {"label_name": label_meta.int2str(example["label"])}


train_hf = train_hf.map(add_label_name)  # try running this line twice
print(train_hf[0]["label_name"])

# streaming=True downloads shards progressively: peek into a dataset
# that would not fit on disk (no len(), no random access)
first = next(
    iter(hf.load_dataset("blanchon/EuroSAT_RGB", split="train", streaming=True))
)
print(first["filename"])


# ------------------------------------- 4. your own files: a custom Dataset
# The whole contract of a Dataset: a __len__ and a __getitem__ returning
# (input, target). Anything with those two methods works with every
# loader and loop in this course. Below is the number-one researcher
# need: images sorted in one folder per class (the layout of section 1).
# torchvision's ImageFolder does this already; writing ours shows how to
# adapt the pattern to messier data.


class ImageFiles(Dataset):
    """Images in one subfolder per class under root; labels from folder names."""

    def __init__(self, root, transform=None, extensions=(".jpg", ".jpeg", ".png")):
        root = Path(root)
        # sorted: alphabetical class names give deterministic, portable
        # label indices (the same convention as ImageFolder)
        self.classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.transform = transform
        # scan the file list once (paths only: no image is read yet);
        # walking class by class gives ImageFolder's order
        self.samples = [
            (path, label)
            for label, name in enumerate(self.classes)
            for path in sorted((root / name).iterdir())
            if path.suffix.lower() in extensions
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        # convert("RGB"): one stray grayscale or RGBA file should not
        # crash a training run
        img = Image.open(path).convert("RGB")
        if self.transform is not None:  # used on the augmentation page
            img = self.transform(img)
        return img, label


mine = ImageFiles(Path("data") / "eurosat" / "2750")
img, label = mine[0]
print(len(mine), "images,", len(mine.classes), "classes")
print(type(img).__name__, img.size, "->", label, mine.classes[label])

# ours agrees with torchvision's reader on this tree:
print(mine.classes == eurosat.classes)  # same classes
print([lbl for _, lbl in mine.samples] == eurosat.targets)  # same labels, same order

# Next page: DataLoaders turn these Datasets into shuffled batches and
# read them in parallel worker processes.
