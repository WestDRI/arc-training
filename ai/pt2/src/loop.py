import math  # for the cosine schedule
import time  # to time epochs
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
from torchvision.datasets import EuroSAT
from torchvision.transforms import v2 as T


class Net(nn.Module):
    """A small CNN for 64x64 images and 10 classes."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # 3 conv blocks: each doubles the channels; the first two halve the image
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),  # 16 -> 16
            nn.AdaptiveAvgPool2d(1),  # one value per channel: 128 features
            nn.Flatten(),
            nn.Linear(128, 10),  # one score per class
        )

    def forward(self, x):
        return self.layers(x)


def main():
    device = (
        torch.accelerator.current_accelerator()
        if torch.accelerator.is_available()
        else torch.device("cpu")
    )
    torch.manual_seed(42)
    print(f"Torch version: {torch.__version__}\nDevice: {device}")

    root = Path("data")

    IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_tf = T.Compose(
        [
            T.ToImage(),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(90),
            T.ToDtype(torch.float32, scale=True),
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

    idx = torch.load(root / "split.pt", weights_only=True)
    train = Subset(EuroSAT(root=root, transform=train_tf), idx["train"])
    val = Subset(EuroSAT(root=root, transform=val_tf), idx["val"])

    loader_kwargs = {
        "batch_size": 64,
        "num_workers": 8,
        "pin_memory": torch.accelerator.is_available(),
    }
    train_loader = DataLoader(
        train,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        **loader_kwargs,
    )
    val_loader = DataLoader(val, **loader_kwargs)
    print(f"Train batches: {len(train_loader)}\nVal batches: {len(val_loader)}")

    model = Net().to(device)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    max_epochs, patience, warmup = 15, 4, len(train_loader)  # warmup: 1 epoch
    total_steps = max_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup  # linear ramp from ~0 to 1
        progress = (step - warmup) / (total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine down to ~0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"Warmup: {warmup} steps\nTotal: {total_steps} steps")

    metric = MulticlassAccuracy(num_classes=10).to(device)
    with torch.inference_mode():
        logits = torch.eye(10).repeat(3, 1).to(device)  # 30 rows of perfect logits
        preds = logits.argmax(dim=1)
    metric.update(preds, torch.arange(10).repeat(3).to(device))
    print(f"Accuracy: {metric.compute().item()}")

    # bf16 only accelerates on Ampere (2020) and newer GPUs; on older ones it
    # runs in emulation and can be slower than plain fp32, hence the gate
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    print(f"Use of mixed precision: {use_bf16}")

    def train_epoch():
        model.train()
        loss_metric = MeanMetric().to(device)
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()  # the schedule advances every batch
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

    def fit():
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
        best_acc, best_epoch, wait = 0.0, 0, 0
        for epoch in range(1, max_epochs + 1):
            t0 = time.perf_counter()
            train_loss = train_epoch()
            val_loss, val_acc = evaluate()
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(scheduler.get_last_lr()[0])
            marker = ""
            if val_acc > best_acc:
                best_acc, best_epoch, wait = val_acc, epoch, 0
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc},
                    root / "best.pt",
                )
                marker = " *"
            else:
                wait += 1
            print(
                f"epoch {epoch:2d}  train {train_loss:.3f}  val {val_loss:.3f}  "
                f"acc {val_acc:5.1%}  lr {scheduler.get_last_lr()[0]:.2e}  "
                f"{time.perf_counter() - t0:4.1f} s{marker}"
            )
            if wait >= patience:
                print(f"early stop: no improvement for {patience} epochs")
                break
        return history, best_epoch, best_acc

    fit()  # returns history, best_epoch, best_acc

    ckpt = torch.load(root / "best.pt", weights_only=True)
    print(f"Best epoch: {ckpt['epoch']} (val acc: {ckpt['val_acc']:.1%})")
    model.load_state_dict(ckpt["model"])
    val_loss, val_acc = evaluate()
    print(f"Restored model: val loss {val_loss:.3f}, acc {val_acc:.1%}")


if __name__ == "__main__":
    main()
