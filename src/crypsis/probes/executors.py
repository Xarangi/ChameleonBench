"""Fit executors: where a refit job's fit runs. Speed only, results identical to a serial fit.

Every executor takes a reads job (crypsis.probes.fit.fit_job's input) and returns a future-like
object whose result() is `((probe or None, info), rng_end_state or None, fit_seconds)`, the
snapshot's refit_overlap contract: fits seed themselves from the probe name, so their result does
not depend on the global torch RNG they start from; the state they END in does, and the caller
restores the serial one at the join (`restore_rng(rec)` when rec is not None).

- SerialExecutor: fits at submit (the reference).
- ThreadExecutor: one worker thread (refit_overlap); stack=True holds stackable rlobf fits and
  trains a kind's held fits as one stacked run (refit_stack_episodes, _StackFitWorker).
- CPUProcessExecutor: spawned worker processes (refit_cpu_fits: pooled kinds only, others go to the
  fallback executor; on_device=True: refit_gpu_procs, every non-detector job is staged on the CPU
  and fit on its own device in the worker). Probes come back on the CPU and are moved to the job's
  device on result() (_OnDevice).
- CUDAStreamExecutor: ThreadExecutor whose fits run on a per-thread side CUDA stream after an
  event recorded on the main stream at submit (refit_overlap_stream). Needs CUDA.

Ported from next_chameleons.tica.trainers.supervised (_iso_fit, _iso_fit_many, _StackFitWorker,
_cpu_fit_init, _cpu_fit_ok, _job_to_cpu, _iso_fit_on, _cpu_tree, _OnDevice). The snapshot's
_cpu_fit_init also copied the main process's depth-rescaled ENS tables into the worker; crypsis has
no depth state (jobs carry absolute layers), so the worker only sets its torch thread count.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Protocol

import torch

from crypsis.probes.fit import fit_job, tok_fit_inputs

_RNG_CANARY = 0x7C3A5EED  # marks "this fit did not touch the global torch RNG"


def _rng_get():
    cu = (torch.cuda.get_rng_state_all()
          if torch.cuda.is_available() and torch.cuda.is_initialized() else None)
    return torch.get_rng_state(), cu


def _rng_set(st) -> None:
    torch.set_rng_state(st[0])
    if st[1] is not None:
        torch.cuda.set_rng_state_all(st[1])


def _rng_eq(a, b) -> bool:
    if not torch.equal(a[0], b[0]) or (a[1] is None) != (b[1] is None):
        return False
    return a[1] is None or all(torch.equal(x, y) for x, y in zip(a[1], b[1], strict=True))


def restore_rng(rec) -> None:
    """Put the global torch RNG where the serial fit left it (rec from a result; None = no-op)."""

    if rec is not None:
        _rng_set(rec)


def ready_dev(dev) -> torch.device:
    return dev if isinstance(dev, torch.device) and dev.type == "cuda" else torch.device(
        "cuda", torch.cuda.current_device())


_SIDE = threading.local()  # one side CUDA stream per worker thread


def iso_fit(job: dict, grad: bool, ready=None):
    """fit_job(job) with the caller's grad mode and device, the global torch RNG isolated.
    Returns ((probe, info), RNG end state or None if untouched, fit seconds). `ready`: a CUDA
    event; the fit then runs on this thread's side stream after it, synchronised before return."""

    t0 = time.time()
    dev = job["device"]
    if isinstance(dev, torch.device) and dev.type == "cuda":
        torch.cuda.set_device(dev)
    before = _rng_get()
    torch.default_generator.manual_seed(_RNG_CANARY)
    if before[1] is not None:
        torch.cuda.manual_seed_all(_RNG_CANARY)
    canary = _rng_get()
    try:
        with torch.set_grad_enabled(grad):
            if ready is not None:
                side = getattr(_SIDE, "s", None)
                if side is None or side.device != ready_dev(dev):
                    side = _SIDE.s = torch.cuda.Stream(device=ready_dev(dev))
                side.wait_event(ready)
                with torch.cuda.stream(side):
                    out = fit_job(job)
                side.synchronize()
            else:
                out = fit_job(job)
        after = _rng_get()
    finally:
        _rng_set(before)
    return out, (None if _rng_eq(after, canary) else after), time.time() - t0


def stackable(job: dict) -> bool:
    """A job ThreadExecutor(stack=True) can stack: an rlobf token fit with refit_batched_rlobf."""

    from crypsis.probes.kinds.zoo import RLOBF_KINDS

    return (job["kind"] in RLOBF_KINDS and job.get("ft") is not None
            and getattr(job["acfg"], "refit_batched_rlobf", False))


def iso_fit_many(jobs: list, grad: bool) -> list:
    """iso_fit for several stackable jobs of ONE kind as one stacked training (gate_fit_many).
    Every job gets the RNG end state of the whole stack."""

    from crypsis.probes.gates import FitPerf, GateConfig, gate_fit_many

    t0 = time.time()
    dev = jobs[0]["device"]
    if isinstance(dev, torch.device) and dev.type == "cuda":
        torch.cuda.set_device(dev)
    before = _rng_get()
    torch.default_generator.manual_seed(_RNG_CANARY)
    if before[1] is not None:
        torch.cuda.manual_seed_all(_RNG_CANARY)
    canary = _rng_get()
    try:
        items = []
        for j in jobs:
            P, N, G = tok_fit_inputs(j)
            items.append({"P": P, "N": N, "NP": G, "name": j["name"], "layer": j["L"],
                          "concept": j["concept"]})
        acfg = jobs[0]["acfg"]
        with torch.set_grad_enabled(grad):
            res = gate_fit_many(jobs[0]["kind"], items, GateConfig.from_refit(acfg), device=dev,
                                C=acfg.refit_C, perf=FitPerf.from_refit(acfg))
        after = _rng_get()
    finally:
        _rng_set(before)
    rec = None if _rng_eq(after, canary) else after
    dt = (time.time() - t0) / len(jobs)
    return [((mon, {**info, "kind": j["full_kind"], "sec_read": j["sec_read"]}), rec, dt)
            for (mon, info), j in zip(res, jobs, strict=True)]


class FitExecutor(Protocol):
    """submit(job, grad=None) -> future (result() = ((probe, info), rng_rec, sec)); flush() before
    waiting on a result; shutdown() when done."""

    def submit(self, job: dict, grad: bool | None = None) -> Any: ...

    def flush(self) -> None: ...

    def shutdown(self) -> None: ...


def _grad(grad):
    return torch.is_grad_enabled() if grad is None else grad


class SerialExecutor:
    """Fits at submit on the calling thread; the reference every other executor must match."""

    def submit(self, job: dict, grad: bool | None = None) -> Future:
        fut: Future = Future()
        try:
            fut.set_result(iso_fit(job, _grad(grad), None))
        except BaseException as e:
            fut.set_exception(e)
        return fut

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class _StackFitWorker:
    """One worker thread; stackable fits are HELD until `max_stack` of a kind are queued or flush()
    is called, then a kind's held fits train as one stacked run; other fits run at once in queue
    order. cancel() works while a fit is queued / held."""

    def __init__(self, max_stack: int = 8):
        self._q: queue.Queue = queue.Queue()
        self._max = max_stack
        self._flush = threading.Event()
        self.n_stacked = 0  # fits that trained inside a stack of >= 2
        threading.Thread(target=self._loop, name="crypsis-refit-stack", daemon=True).start()

    def submit(self, fn, job, grad, ready):
        fut: Future = Future()
        self._q.put((fn, job, grad, ready, fut))
        return fut

    def flush(self):
        self._flush.set()
        self._q.put(None)  # wake the worker

    @staticmethod
    def _run_one(it):
        if it[4].set_running_or_notify_cancel():
            try:
                it[4].set_result(it[0](it[1], it[2], it[3]))
            except BaseException as e:
                it[4].set_exception(e)

    def _run_group(self, g):
        g = [it for it in g if it[4].set_running_or_notify_cancel()]
        if not g:
            return
        if len(g) == 1:
            try:
                g[0][4].set_result(g[0][0](g[0][1], g[0][2], g[0][3]))
            except BaseException as e:
                g[0][4].set_exception(e)
            return
        try:
            outs = iso_fit_many([it[1] for it in g], g[0][2])
            self.n_stacked += len(g)
            for it, o in zip(g, outs, strict=True):
                it[4].set_result(o)
        except BaseException as e:
            for it in g:
                it[4].set_exception(e)

    def _loop(self):
        held: dict = {}
        while True:
            items = [self._q.get()]
            while True:
                try:
                    items.append(self._q.get_nowait())
                except queue.Empty:
                    break
            for it in items:
                if it is None:
                    continue
                if it[3] is None and stackable(it[1]):
                    held.setdefault(it[1]["kind"], []).append(it)
                else:
                    self._run_one(it)
            flush = self._flush.is_set()
            for kind in list(held):
                if flush or len(held[kind]) >= self._max:
                    self._run_group(held.pop(kind))
            if flush and not held:
                self._flush.clear()


class ThreadExecutor:
    """One worker thread (refit_overlap); stack=True: the stacking worker (refit_stack_episodes)."""

    def __init__(self, stack: bool = False, max_stack: int = 8) -> None:
        self._pool = (_StackFitWorker(max_stack) if stack
                      else ThreadPoolExecutor(max_workers=1, thread_name_prefix="crypsis-refit"))

    @property
    def n_stacked(self) -> int:
        return getattr(self._pool, "n_stacked", 0)

    def _submit(self, job, grad, ready):
        return self._pool.submit(iso_fit, job, _grad(grad), ready)

    def submit(self, job: dict, grad: bool | None = None):
        return self._submit(job, grad, None)

    def flush(self) -> None:
        if hasattr(self._pool, "flush"):
            self._pool.flush()

    def shutdown(self) -> None:
        if isinstance(self._pool, ThreadPoolExecutor):
            self._pool.shutdown(wait=True)


class CUDAStreamExecutor(ThreadExecutor):
    """ThreadExecutor whose fits run on a side CUDA stream after the main stream's queued reads."""

    def __init__(self, stack: bool = False, max_stack: int = 8) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAStreamExecutor needs CUDA")
        super().__init__(stack, max_stack)

    def submit(self, job: dict, grad: bool | None = None):
        ready = torch.cuda.Event()
        ready.record()
        return self._submit(job, grad, ready)


# ----------------------------------------------------------------------------- worker processes
def _cpu_fit_init(threads: int) -> None:
    torch.set_num_threads(max(1, int(threads)))


def cpu_fit_ok(job: dict) -> bool:
    """A job whose fit can run in a CPU worker: pooled kinds (no per-token states, no detector)."""

    from crypsis.probes.kinds.evdet import KINDS as DETECTOR_KINDS

    return (job["kind"] not in DETECTOR_KINDS and job.get("ft") is None
            and job.get("nt") is None)


def _tree(x, f):
    if isinstance(x, torch.Tensor):
        return f(x)
    if isinstance(x, dict):
        return {k: _tree(v, f) for k, v in x.items()}
    if isinstance(x, list):
        return [_tree(v, f) for v in x]
    return x


def job_to_cpu(job: dict) -> dict:
    """A copy of a job with every tensor on the CPU and device = cpu."""

    return {**{k: _tree(v, lambda t: t.detach().cpu()) for k, v in job.items()},
            "device": torch.device("cpu")}


def _cpu_tree(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_cpu_tree(v) for v in x)
    return x


def iso_fit_on(job: dict, grad: bool):
    """Worker of on_device processes: the job arrives staged on the CPU with its original device;
    move it there, fit (iso_fit), send the probe back on the CPU (no CUDA IPC)."""

    dev = job["device"]
    (mon, info), rec, sec = iso_fit(
        {**{k: _tree(v, lambda t: t.to(dev)) for k, v in job.items()}, "device": dev}, grad, None)
    return (None if mon is None else mon.to("cpu"), _cpu_tree(info)), rec, sec


class OnDevice:
    """A worker-process fit's future: result() moves the probe to the device a serial fit would
    have left it on."""

    def __init__(self, fut, dev):
        self._f, self._dev = fut, dev

    def result(self):
        (mon, info), rec, sec = self._f.result()
        return (None if mon is None else mon.to(self._dev), info), rec, sec

    def cancel(self):
        return self._f.cancel()

    def cancelled(self):
        return self._f.cancelled()

    def done(self):
        return self._f.done()


class CPUProcessExecutor:
    """Spawned worker processes. on_device=False (refit_cpu_fits): pooled-kind jobs fit on the CPU
    in a worker, others go to `fallback`. on_device=True (refit_gpu_procs): every non-detector job
    fits in a worker on its own device."""

    def __init__(self, workers: int, threads: int = 4, *, on_device: bool = False,
                 fallback: FitExecutor | None = None) -> None:
        self._pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                         initializer=_cpu_fit_init, initargs=(threads,))
        self._on_device = on_device
        self._fallback = fallback or SerialExecutor()

    def submit(self, job: dict, grad: bool | None = None):
        from crypsis.probes.kinds.evdet import KINDS as DETECTOR_KINDS

        grad = _grad(grad)
        if self._on_device and job["kind"] not in DETECTOR_KINDS:
            staged = {**job_to_cpu(job), "device": job["device"]}
            return OnDevice(self._pool.submit(iso_fit_on, staged, grad), job["device"])
        if not self._on_device and cpu_fit_ok(job):
            return OnDevice(self._pool.submit(iso_fit, job_to_cpu(job), grad, None),
                            job["device"])
        return self._fallback.submit(job, grad)

    def flush(self) -> None:
        self._fallback.flush()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
        self._fallback.shutdown()
