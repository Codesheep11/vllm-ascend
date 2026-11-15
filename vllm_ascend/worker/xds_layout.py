# SPDX-License-Identifier: Apache-2.0
"""Final-page padding for xDS; padding is stored with KV, not a staging buffer."""

import math

import torch

XDS_IO_ALIGNMENT = 4096


def scale_padding_width(block_size: int, dtype: torch.dtype) -> int:
    """Return the minimum width making a scalar scale page I/O aligned."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    page_bytes = block_size * torch.empty((), dtype=dtype, device="cpu").element_size()
    return XDS_IO_ALIGNMENT // math.gcd(page_bytes, XDS_IO_ALIGNMENT)


def scale_cache_for_indexer(cache: torch.Tensor) -> torch.Tensor:
    """Expose page padding as a physical token stride for LightningIndexerQuant.

    Logical scales occupy the first block_size elements of each physical page.
    The patched kernel reads only those elements, using the full page stride.
    This is a view and does not copy the cache.
    """
    return cache.view(cache.shape[0], -1, 1)


def scale_slot_mapping(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Map logical token slots to padded scale pages, preserving invalid slots."""
    if cache.shape[-1] == 1:
        return slots
    block_size = cache.shape[1]
    mapped = (slots // block_size) * (block_size * cache.shape[-1]) + slots % block_size
    return torch.where(slots < 0, slots, mapped)
