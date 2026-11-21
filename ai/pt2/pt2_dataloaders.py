"""Draft of the code for the DataLoaders page (pt2_dataloaders.qmd).

DataLoaders turn Datasets into shuffled batches of tensors and read
them in parallel worker processes. Continuing from pt2_datasets.qmd:
EuroSAT is on disk in ./data and the train/validation split is saved
as data/split.pt.

Everything runs on CPU (the GPU timing demo of the page is a static
block measured beforehand on an RTX 2060).

Python 3.14 note: the default multiprocessing start method on Linux is
now forkserver, so a plain script that spawns DataLoader workers must
put all its code under if __name__ == "__main__": (notebook cells are
immune: an interactive __main__ cannot be re-imported). Verified:
module-level code outside the guard re-runs in the forkserver process
at worker startup.

Run with: uv run python pt2_dataloaders.py
"""

import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT
from torchvision.transforms import v2 as T


def one_pass(dataset, num_workers):
    """Time one full pass over a dataset (21,600 images)."""
    loader = DataLoader(dataset, batch_size=64, num_workers=num_workers)
    n = 0
    t0 = time.perf_counter()
    for _, labels in loader:
        n += labels.numel()
    assert n == len(dataset)
    return time.perf_counter() - t0


def second_epoch_cost(dataset, persistent):
    """Cost of the first batch of epoch 2 (worker restart or reuse)."""
    loader = DataLoader(
        dataset, batch_size=64, num_workers=4, persistent_workers=persistent
    )
    it = iter(loader)  # epoch 1: spawn workers
    next(it)  # first batch (decode + transform)
    del it  # end of epoch 1: workers die (or persist)
    t0 = time.perf_counter()
    it = iter(loader)  # epoch 2: respawn (or reuse)
    next(it)
    cost = time.perf_counter() - t0
    del it
    return cost


def main():
    # --------------------------------------- 1. from Dataset to batches
    # Reload the dataset and the exact split saved on the datasets
    # page: every later page must evaluate on this validation set.
    eurosat = EuroSAT(root="data")
    idx = torch.load("data/split.pt", weights_only=True)
    train = Subset(eurosat, idx["train"])
    val = Subset(eurosat, idx["val"])
    print(len(train), len(val))

    # Indexing still returns ONE sample at a time (a PIL image + an
    # int).
    img, label = train[0]
    print(type(img).__name__, label)

    # A DataLoader assembles batches, but the default collate function
    # can only stack tensors, arrays, and numbers: on PIL images it
    # crashes.
    try:
        next(iter(DataLoader(train, batch_size=64)))
    except TypeError as e:
        print(e)

    # A model wants one float tensor of shape (batch, channels,
    # height, width). The fix is a transform attached to the dataset.
    # ToImage() converts PIL -> uint8 CHW tensor; ToDtype(scale=True)
    # divides by 255 into float32 in [0, 1]. The full preprocessing
    # pipeline is the topic of the next page; this minimal pair is all
    # we need for batches.
    to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

    eurosat = EuroSAT(root="data", transform=to_tensor)
    train = Subset(eurosat, idx["train"])
    val = Subset(eurosat, idx["val"])

    images, labels = next(iter(DataLoader(train, batch_size=64)))
    print(images.shape, images.dtype, images.min().item(), images.max().item())
    print(labels.shape, labels.dtype)

    # number of batches: ceil(21600 / 64) = 338 (the last batch has 32
    # images); with drop_last=True, incomplete batches are dropped
    print(len(DataLoader(train, batch_size=64)))
    print(len(DataLoader(train, batch_size=64, drop_last=True)))

    # ----------------------------------------------- 2. shuffle, and why
    # The files on disk are stored class by class (AnnualCrop first,
    # then Forest...). An unshuffled loader over the raw dataset feeds
    # nearly single-class batches.
    plain = EuroSAT(root="data", transform=to_tensor)
    first = next(iter(DataLoader(plain, batch_size=8, shuffle=False)))[1]
    second = next(iter(DataLoader(plain, batch_size=8, shuffle=False)))[1]
    print(first.tolist())
    print(second.tolist())

    # shuffle=True draws a fresh order every epoch...
    l1 = next(iter(DataLoader(train, batch_size=8, shuffle=True)))[1]
    l2 = next(iter(DataLoader(train, batch_size=8, shuffle=True)))[1]
    print(l1.tolist())
    print(l2.tolist())

    # ...and a seeded generator makes the order reproducible
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = next(iter(DataLoader(train, batch_size=8, shuffle=True, generator=g1)))[1]
    b = next(iter(DataLoader(train, batch_size=8, shuffle=True, generator=g2)))[1]
    print(torch.equal(a, b))

    # -------------------------------------------------- 3. num_workers
    # num_workers=0 (default): samples are read, decoded, and
    # transformed in the main process, one at a time, while the
    # accelerator waits. With n workers, n processes decode and
    # transform in parallel and hand ready batches over through a
    # queue. Warm up first: process startup is paid once, and we want
    # steady-state numbers (first-epoch cost is the next section).
    one_pass(train, 2)
    for w in (0, 2, 4, 8):
        print(f"num_workers={w}: {one_pass(train, w):6.1f} s")

    # --------------------------- 4. prefetch, persistent workers
    # Each worker prepares prefetch_factor (default 2) batches ahead
    # of what the main process consumes: loading overlaps with
    # compute. Only in effect when num_workers > 0.

    # Workers are spawned when iteration starts and die when the
    # iterator is exhausted: every epoch pays the startup cost again.
    # persistent_workers=True keeps them alive across epochs. Cost of
    # the first batch of epoch 2, with and without:
    print(f"second epoch, default:    {second_epoch_cost(train, False):.2f} s")
    print(f"second epoch, persistent: {second_epoch_cost(train, True):.2f} s")

    # ----------------------------------------------------- 5. pin_memory
    # Host-to-device copies go faster from page-locked ("pinned")
    # memory and can run asynchronously (non_blocking=True) while the
    # previous step computes. pin_memory=True pins batches in the main
    # process; it only matters when training on an accelerator. Set it
    # conditionally:
    print(torch.accelerator.is_available())


if __name__ == "__main__":
    main()
