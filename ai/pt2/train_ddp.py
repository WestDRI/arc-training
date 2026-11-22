"""Train the course's small CNN on EuroSAT across several accelerators.

Distributed version of train.py: torchrun sets up the process group, each
process (rank) drives one accelerator, DistributedSampler splits the data,
DDP averages the gradients across ranks at every step.

CPU smoke test (2 processes, no GPU needed):
    torchrun --nproc-per-node 2 train_ddp.py --epochs 1 --max-batches 3

Single node, N GPUs:
    torchrun --nproc-per-node N train_ddp.py --epochs 15
"""

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision.datasets import EuroSAT
from torchvision.transforms import v2 as T

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class Net(nn.Module):
    """A small CNN for 64x64 images and 10 classes."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.layers(x)


def all_reduce_mean(value, device):
    """Average a Python float across all ranks."""
    t = torch.tensor([value], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--data", default="data", help="data directory")
    parser.add_argument("--out", default=None, help="checkpoint path")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="truncate epochs (smoke tests only)")
    parser.add_argument("--cpu", action="store_true",
                        help="force CPU + gloo (smoke tests without GPUs)")
    args = parser.parse_args()
    out = args.out or f"{args.data}/best_ddp.pt"

    # --- distributed setup (torchrun provides all of these) ---------------
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    use_accelerator = torch.accelerator.is_available() and not args.cpu
    backend = "nccl" if use_accelerator else "gloo"
    dist.init_process_group(backend=backend)
    device = torch.device("cuda", local_rank) if use_accelerator \
        else torch.device("cpu")
    is_main = rank == 0
    if is_main:
        print(f"torch {torch.__version__}  world size {world_size}  "
              f"backend {backend}", flush=True)

    torch.manual_seed(42)  # same seed on every rank: identical model init

    # --- data: each rank gets its own shard of every epoch ----------------
    train_tf = T.Compose([
        T.ToImage(),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(90),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    idx = torch.load(f"{args.data}/split.pt", weights_only=True)
    train = Subset(EuroSAT(root=args.data, transform=train_tf), idx["train"])
    val = Subset(EuroSAT(root=args.data, transform=val_tf), idx["val"])
    if args.max_batches:
        train = Subset(train, range(args.max_batches * 64 * world_size))
        val = Subset(val, range(256))

    sampler = DistributedSampler(train, shuffle=True, seed=42)
    train_loader = DataLoader(
        train, batch_size=64, sampler=sampler, num_workers=2,
        persistent_workers=True, pin_memory=use_accelerator,
    )
    # exact partition of validation across ranks (no padding, no overlap)
    val = Subset(val, range(rank, len(val), world_size))
    val_loader = DataLoader(
        val, batch_size=128, num_workers=2,
        persistent_workers=True, pin_memory=use_accelerator,
    )

    # --- model: wrapped in DDP, gradients averaged across ranks -----------
    model = Net().to(device)
    ddp_model = DDP(
        model, device_ids=[local_rank] if device.type == "cuda" else None
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    warmup = len(train_loader)
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / (total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def evaluate():
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.inference_mode():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = ddp_model(x)
                loss_sum += loss_fn(logits, y).item() * x.size(0)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += x.size(0)
        # every rank evaluated its shard: combine into global numbers
        correct = all_reduce_mean(correct / total, device) * total
        loss_sum = all_reduce_mean(loss_sum / total, device) * total
        return loss_sum / total, correct / total

    best_acc, wait = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)  # fresh shard assignment every epoch
        t0 = time.perf_counter()
        model.train()
        train_loss, seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = loss_fn(ddp_model(x), y)
            loss.backward()  # DDP hooks in: gradients averaged across ranks
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * x.size(0)
            seen += x.size(0)
        train_loss = all_reduce_mean(train_loss / seen, device)
        val_loss, val_acc = evaluate()
        if is_main:
            marker = ""
            if val_acc > best_acc:
                best_acc, wait = val_acc, 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc,
                    },
                    out,
                )
                marker = " *"
            else:
                wait += 1
            print(f"epoch {epoch:2d}  train {train_loss:.3f}  "
                  f"val {val_loss:.3f}  acc {val_acc:5.1%}  "
                  f"{time.perf_counter() - t0:4.1f} s{marker}", flush=True)
        else:
            if val_acc > best_acc:  # keep ranks in step for early stopping
                best_acc, wait = val_acc, 0
            else:
                wait += 1
        if wait >= args.patience:
            if is_main:
                print(f"early stop: no improvement for {args.patience} epochs",
                      flush=True)
            break

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
