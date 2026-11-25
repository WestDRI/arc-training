"""Draft of the code for the preprocessing & augmentation page
(pt2_augmentation.qmd).

Transforms v2 turn PIL images into normalized tensors (preprocessing)
and randomly perturb the training images (augmentation). Continuing
from pt2_datasets.qmd and pt2_dataloaders.qmd: EuroSAT in ./data,
split saved as data/split.pt.

Everything runs on CPU. Run with: uv run python pt2_augmentation.py
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT
from torchvision.transforms import v2 as T

torch.manual_seed(42)


def main():

    # ------------------------------------------------- 1. the two pipelines
    # Preprocessing (deterministic, applied to every split) turns images
    # into normalized float tensors. Augmentation (random, training only)
    # perturbs images in ways that do not change their label.

    # ImageNet statistics: the mean/std the pretrained models of Part 3
    # were normalized with; we adopt them course-wide so results stay
    # comparable from the scratch baseline to fine-tuning.
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_tf = T.Compose(
        [
            T.ToImage(),  # PIL -> uint8 CHW tensor
            T.RandomHorizontalFlip(),  # p=0.5 by default
            T.RandomVerticalFlip(),  # p=0.5 by default
            T.RandomRotation(90),  # random angle in [-90, 90]
            T.ToDtype(torch.float32, scale=True),  # /255 -> float32 in [0, 1]
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    val_tf = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # why no resize? EuroSAT images are all 64x64 already. For mixed-size
    # data, val gets Resize/CenterCrop and train gets RandomResizedCrop.

    # --------------------------------------- 2. the pipeline on a raw image
    eurosat = EuroSAT(root="data")  # plain instance: raw PIL access
    idx = torch.load("data/split.pt", weights_only=True)

    img, label = eurosat[idx["train"][0]]
    print(type(img).__name__, img.size, img.mode, "->", label, eurosat.classes[label])

    # stage by stage on this one image:
    x = T.ToImage()(img)
    print(x.shape, x.dtype, int(x.min()), int(x.max()))
    x = T.ToDtype(torch.float32, scale=True)(x)
    print(x.shape, x.dtype, f"{x.min():.3f}", f"{x.max():.3f}")
    x = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)
    print(x.shape, x.dtype, f"{x.min():.2f}", f"{x.max():.2f}")
    print(
        "per-channel means after Normalize:",
        [f"{v:+.2f}" for v in x.mean(dim=(1, 2)).tolist()],
    )

    # ------------------------------------------- 3. random vs deterministic
    # validation transform: same image in, same tensor out
    a, b = val_tf(img), val_tf(img)
    print("val deterministic:", torch.equal(a, b))

    # training transform: same image in, a different variant every call
    c, d, e = train_tf(img), train_tf(img), train_tf(img)
    print("train varies:", not torch.equal(c, d), not torch.equal(d, e))

    # over 5 epochs of 1 batch-step... point: the model never sees the
    # same tensor twice even though the file count does not change

    # ------------------------------------------------- 4. batched warning
    # v2 transforms accept batches, but they draw ONE set of random
    # parameters per call: the whole batch gets the same flip/rotation.
    # Augment per sample (in the Dataset), not on batches.
    batch = torch.stack([T.ToImage()(eurosat[i][0]) for i in range(8)])
    hflip = T.RandomHorizontalFlip()
    for _ in range(5):
        out = hflip(batch)
        flipped = [not torch.equal(out[i], batch[i]) for i in range(len(batch))]
        print("flipped per image:", flipped)  # always all-True or all-False

    # --------------------------------------------- 5. the assembled loaders
    # transforms run inside __getitem__, i.e. in the worker processes of
    # the previous page: augmentation is parallelized for free.
    train = Subset(EuroSAT(root="data", transform=train_tf), idx["train"])
    val = Subset(EuroSAT(root="data", transform=val_tf), idx["val"])
    print(len(train), len(val))

    loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=4)
    x, y = next(iter(loader))
    print(x.shape, x.dtype, f"{x.min():.2f}", f"{x.max():.2f}")
    print(y.shape, y.dtype)


if __name__ == "__main__":
    main()
