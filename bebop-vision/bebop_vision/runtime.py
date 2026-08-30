"""Shared torch runtime helpers."""

import contextlib

import torch


def resolve_device(device="auto"):
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def autocast_ctx(device):
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()