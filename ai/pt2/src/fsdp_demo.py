"""A minimal FSDP2 demonstration: shard a model, take a step, save sharded.

FSDP matters when a model does not fit on one GPU. Our course CNN does fit,
so this demo uses a synthetic model to show the mechanics -- the sharding of
parameters, the sharded optimizer state, and a distributed checkpoint --
without any training time. Run it on CPUs, no GPU required:

    torchrun --nproc-per-node 2 fsdp_demo.py
"""

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint import FileSystemWriter, save
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc = nn.Linear(width, width)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.fc(x))


class Big(nn.Module):
    """Stand-in for a model too large for one GPU."""

    def __init__(self, width=1024, depth=8):
        super().__init__()
        self.blocks = nn.ModuleList([Block(width) for _ in range(depth)])
        self.head = nn.Linear(width, 10)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def main():
    dist.init_process_group(backend="gloo")  # nccl on GPUs
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    mesh = init_device_mesh("cpu", (world_size,))
    is_main = rank == 0

    torch.manual_seed(0)  # identical model on every rank before sharding
    model = Big()
    n_full = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"model created: {n_full:,} parameters")

    # --- shard it: children first, root last --------------------------------
    for block in model.blocks:
        fully_shard(block, mesh=mesh)
    fully_shard(model.head, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # parameters are now DTensors: global view from here, shard on disk
    n_local = sum(p.to_local().numel() for p in model.parameters())
    if is_main:
        print(f"parameters visible per rank after sharding: "
              f"{n_full:,} (global shape)")
        print(f"parameters actually stored per rank: "
              f"{n_local:,} (~1/{round(n_full / n_local)} of the model)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # --- one sharded training step ------------------------------------------
    x = torch.randn(8, 1024)
    y = torch.randint(0, 10, (8,))
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
    dist.barrier()
    if is_main:
        print(f"one sharded step OK, loss {loss.item():.3f}")

    # --- a distributed checkpoint: each rank writes its own shards ----------
    state = {"model": get_model_state_dict(model)}
    save(state, storage_writer=FileSystemWriter("fsdp_ckpt"))
    dist.barrier()
    if is_main:
        print("distributed checkpoint written to fsdp_ckpt/")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
