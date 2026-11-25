import time  # to time experiments
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT
from torchvision.transforms import v2 as T

device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)

to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
root = Path("data")
eurosat = EuroSAT(root=root, transform=to_tensor)
idx = torch.load(root / "split.pt", weights_only=True)
train = Subset(eurosat, idx["train"])

for num_workers in (0, 2, 4, 8, 16):
    loader = DataLoader(
        train, batch_size=64, shuffle=True, pin_memory=True, num_workers=num_workers
    )
    for epoch in range(2):  # warmup, then the timed epoch
        t0 = time.perf_counter()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch:  # mean_util: average of the nvidia-smi samples
            print(
                f"num_workers={num_workers}: "
                f"{time.perf_counter() - t0:5.1f} s/epoch, GPU util {mean_util:3.0f}%"
            )
