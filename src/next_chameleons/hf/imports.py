"""Lazy optional dependency imports for real HF runs."""

from __future__ import annotations


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install real-run dependencies with `uv sync --extra ml`.") from exc
    return torch


def require_datasets():
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("Install real-run dependencies with `uv sync --extra ml`.") from exc
    return datasets


def require_transformers():
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError("Install real-run dependencies with `uv sync --extra ml`.") from exc
    return transformers


def require_peft():
    try:
        import peft
    except ImportError as exc:
        raise RuntimeError("Install real-run dependencies with `uv sync --extra ml`.") from exc
    return peft

