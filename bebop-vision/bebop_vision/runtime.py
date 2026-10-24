"""Shared torch runtime helpers."""

import contextlib

import torch


def autocast_ctx(device):
    if device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
