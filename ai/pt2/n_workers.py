import time  # to time experiments
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT
from torchvision.transforms import v2 as T

to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
root = Path("data")
eurosat = EuroSAT(root=root, transform=to_tensor)
idx = torch.load(root / "split.pt", weights_only=True)
train = Subset(eurosat, idx["train"])


def timed_pass(num_workers):
    """Time one full pass over the training data at steady state."""
    loader = DataLoader(train, batch_size=64, num_workers=num_workers)
    for _ in loader:
        pass  # warmup
    t0 = time.perf_counter()
    n = 0
    for _, labels in loader:
        n += labels.numel()
    dt = time.perf_counter() - t0
    del loader  # shuts the workers down
    assert n == len(train)
    return dt


if __name__ == "__main__":
    for w in (0, 2, 4, 8, 16):
        print(f"num_workers={w}: {timed_pass(w):5.1f} s")
