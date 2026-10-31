"""Draft of the code for the intro page (pt2_intro.qmd).

A fast refresher of the essentials in one running example (two synthetic
spirals): tensors, nn.Module, forward, loss, backward, step, then the bare
loop that strings them together. Runs on CPU in seconds.

Run with: uv run python intro.py
"""

import torch
from torch import nn

# Accelerator if there is one, CPU otherwise
device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)
print(f"torch {torch.__version__}, {device}\n")

torch.manual_seed(42)  # reproducible toy data and initialization

# ---------------------------------------------------------------- 1. tensors
# PyTorch's fundamental object: an n-dimensional array with a shape,
# a dtype, and a device. NumPy users are on familiar ground.

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # from Python data
print(x)
y = x.to(device)  # move to GPU
print(y)
w = torch.randn(2, 5, device=device)  # random, straight on the accelerator
print(y @ w)  # operators, as you would expect (need to be on the same device)


def two_spirals(n_per_class=500, noise=0.15):
    """Two interleaved spirals: our toy dataset for this page."""
    t = torch.rand(n_per_class, 1).sqrt() * (3 * torch.pi)
    arm = torch.hstack([t * torch.cos(t), t * torch.sin(t)])
    arm1 = arm + torch.randn(n_per_class, 2) * noise
    arm2 = -arm + torch.randn(n_per_class, 2) * noise
    x = torch.vstack([arm1, arm2]).to(device)
    y = torch.cat(
        [
            torch.zeros(n_per_class, dtype=torch.long),
            torch.ones(n_per_class, dtype=torch.long),
        ]
    ).to(device)
    return x, y


x, y = two_spirals()  # the data are tensors too: 1000 points, 2 classes
print(x.shape, y.shape)


# ------------------------------------------------------------- 2. nn.Module
# Networks are Python classes subclassing nn.Module. Layers are created
# in __init__, wired in forward. Widths in powers of two.


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.layers(x)


model = Net().to(device)
print(model)
print("parameters:", sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------- 3. forward
# Calling the model runs forward: 1000 points in, one row of 2 raw
# scores (logits) per point out.

logits = model(x)
print(logits[:4])
print(logits.softmax(dim=-1)[:2])  # probabilities, for inspection only


# ------------------------------------------------------------------ 4. loss
# The loss measures how wrong the model is. CrossEntropyLoss takes
# logits (not probabilities) and integer class labels.

criterion = nn.CrossEntropyLoss()
loss = criterion(logits, y)
print(loss.item())


# ---------------------------------------------------------------- 5. backward
# loss.backward() runs autograd: every parameter tensor that influenced
# the loss gets its gradient in .grad.

print(model.layers[0].weight.grad)  # None: nothing computed yet
loss.backward()
g = model.layers[0].weight.grad
print(g.shape, g.norm().item())


# -------------------------------------------------------------------- 6. step
# The optimizer updates every parameter against its gradient.

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
optimizer.step()  # nudge the parameters
optimizer.zero_grad()  # gradients accumulate: clear them after each step

with torch.inference_mode():  # no autograd bookkeeping when evaluating
    print(criterion(model(x), y).item())


# ------------------------------------------------------------- 7. bare loop
# String it together: zero the gradients, forward, loss, backward, step.
# This is the whole skeleton of deep learning training. Part 2 builds
# the modern version (bf16, warmup + cosine, torch.compile, early
# stopping, metrics) around it; Part 3 reuses it unchanged on a
# pretrained model.

for epoch in range(1, 501):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

with torch.inference_mode():
    acc = (model(x).argmax(dim=1) == y).float().mean()
print(f"training accuracy: {acc:.0%}")
