"""Train the course's small CNN on EuroSAT (single accelerator).

Script version of the training loop of ai/pt2/pt2_loop.qmd: same model,
same schedule, plus command-line arguments and a resumable checkpoint.

Tested with:  python train.py --epochs 1 --max-batches 5
Typical run:  python train.py --epochs 15
"""

import argparse
import math
import time

import torch
from torch import nn
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.data import DataLoader, Subset
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


def make_loaders(data_dir, max_batches=None):
    """Canonical loaders; with max_batches, epochs cover a subset (smoke tests)."""
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
    idx = torch.load(f"{data_dir}/split.pt", weights_only=True)
    train = Subset(EuroSAT(root=data_dir, transform=train_tf), idx["train"])
    val = Subset(EuroSAT(root=data_dir, transform=val_tf), idx["val"])
    if max_batches:
        # shrink the datasets so one epoch is a handful of batches
        train = Subset(train, range(max_batches * 64))
        val = Subset(val, range(128))
    loader_kwargs = {
        "batch_size": 64,
        "num_workers": 4,
        "persistent_workers": True,
        "pin_memory": torch.accelerator.is_available(),
    }
    train_loader = DataLoader(
        train, shuffle=True, generator=torch.Generator().manual_seed(42),
        **loader_kwargs,
    )
    val_loader = DataLoader(val, **loader_kwargs)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--data", default="data", help="data directory")
    parser.add_argument("--out", default=None, help="checkpoint path")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="truncate epochs (smoke tests only)")
    args = parser.parse_args()
    out = args.out or f"{args.data}/best_script.pt"

    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available()
        else torch.device("cpu")
    )
    torch.manual_seed(42)
    print(f"torch {torch.__version__}  device {device}")

    train_loader, val_loader = make_loaders(args.data, args.max_batches)

    model = Net().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    warmup = len(train_loader)  # 1 epoch in steps
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / (total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def train_epoch():
        model.train()
        loss_metric = MeanMetric().to(device)
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_metric.update(loss.detach(), weight=x.size(0))
        return loss_metric.compute().item()

    def evaluate():
        model.eval()
        loss_metric = MeanMetric().to(device)
        acc_metric = MulticlassAccuracy(num_classes=10).to(device)
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.inference_mode():
                logits = model(x)
                preds = logits.argmax(dim=1)
            loss_metric.update(loss_fn(logits, y), weight=x.size(0))
            acc_metric.update(preds, y)
        return loss_metric.compute().item(), acc_metric.compute().item()

    best_acc, wait = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_epoch()
        val_loss, val_acc = evaluate()
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
        print(f"epoch {epoch:2d}  train {train_loss:.3f}  val {val_loss:.3f}  "
              f"acc {val_acc:5.1%}  {time.perf_counter() - t0:4.1f} s{marker}",
              flush=True)
        if wait >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs")
            break


if __name__ == "__main__":
    main()
