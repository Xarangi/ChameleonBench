"""Data-parallel episodes for the adaptive trainer (REPLICATION 2026-10-19b; default off).

N processes, one GPU each, each run episodes_per_step / N of the step's episodes (rank r: episodes
r, r + N, ...), refits included (the dominant cost). The episode specs are sampled identically on
every rank (same sampler RNG); every in-rollout draw comes from a per-episode stream
(adaptive.episode_rng), so an episode does the same thing whichever rank runs it. After the rollout
the ranks all-gather their episode records and apply the same deterministic updates (curriculum
bandits, pursuit centroid, trainer RNG), gradients are summed with an all-reduce (the loss is
normalised by step totals, which are all-reduced first) and every rank takes the same optimizer
step. Rank 0 alone logs, checkpoints and runs the capability terms (cap_gen / retain anchor) on
the gathered episodes. Launch: scripts/slurm/tica_l3covert.sh with TICA_DP=N (one process per GPU).
"""

from __future__ import annotations

import os

import torch


def _dist():
    import torch.distributed as dist

    return dist if dist.is_available() and dist.is_initialized() else None


def world() -> tuple[int, int]:
    """(world size, rank); (1, 0) without an initialised process group."""

    d = _dist()
    return (d.get_world_size(), d.get_rank()) if d is not None else (1, 0)


def is_main() -> bool:
    return world()[1] == 0


def init_from_env() -> bool:
    """Initialise the process group from WORLD_SIZE / RANK / MASTER_ADDR / MASTER_PORT (set by the
    launcher). NCCL when CUDA is available (each process sees ONE GPU: the launcher sets
    CUDA_VISIBLE_DEVICES per rank), else gloo. Returns True if data-parallel."""

    n = int(os.environ.get("WORLD_SIZE", "1") or 1)
    if n <= 1:
        return False
    import torch.distributed as dist

    if not dist.is_initialized():
        from datetime import timedelta

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if backend == "nccl":
            torch.cuda.set_device(0)
        # generous: at start-up rank 0 alone fits the base probes / generates the anchor answers
        # (uncached: tens of minutes) while the other ranks wait in a barrier
        dist.init_process_group(backend, timeout=timedelta(
            seconds=int(os.environ.get("TICA_DP_TIMEOUT", "7200"))))
    return True


def gather(obj) -> list:
    """all_gather_object: [obj of rank 0, obj of rank 1, ...] on every rank."""

    d = _dist()
    if d is None:
        return [obj]
    out = [None] * d.get_world_size()
    d.all_gather_object(out, obj)
    return out


def broadcast(obj, src: int = 0):
    d = _dist()
    if d is None:
        return obj
    box = [obj]
    d.broadcast_object_list(box, src=src)
    return box[0]


def barrier() -> None:
    d = _dist()
    if d is not None:
        d.barrier()


def _device() -> torch.device:
    d = _dist()
    if d is not None and d.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def sum_floats(xs: list[float]) -> list[float]:
    """Element-wise sum over ranks (float64)."""

    d = _dist()
    if d is None:
        return list(xs)
    t = torch.tensor(xs, dtype=torch.float64, device=_device())
    d.all_reduce(t)
    return t.tolist()


def allreduce_grads(params) -> None:
    """Sum the parameters' gradients over ranks (missing grads count as zero), one flat buffer per
    dtype / device."""

    d = _dist()
    if d is None:
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    groups: dict = {}
    for p in params:
        groups.setdefault((p.grad.dtype, p.grad.device), []).append(p)
    for ps in groups.values():
        flat = torch.cat([p.grad.reshape(-1) for p in ps])
        d.all_reduce(flat)
        off = 0
        for p in ps:
            n = p.grad.numel()
            p.grad.copy_(flat[off : off + n].view_as(p.grad))
            off += n


def allreduce_tensors(ts) -> None:
    """In place: sum the tensors over ranks, one flat buffer per dtype / device (the hide-grad
    buffer of train.hide_grad_clip, 10-21r)."""

    d = _dist()
    if d is None:
        return
    groups: dict = {}
    for t in ts:
        groups.setdefault((t.dtype, t.device), []).append(t)
    for g in groups.values():
        flat = torch.cat([t.reshape(-1) for t in g])
        d.all_reduce(flat)
        off = 0
        for t in g:
            n = t.numel()
            t.copy_(flat[off : off + n].view_as(t))
            off += n


def params_checksum(params) -> list[float]:
    """Per-rank float64 sum of all parameters (drift check: equal on every rank)."""

    s = float(sum(float(p.detach().double().sum()) for p in params))
    return [float(x) for x in gather(s)]
