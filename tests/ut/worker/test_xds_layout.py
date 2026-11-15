# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.worker.xds_layout import scale_cache_for_indexer, scale_padding_width, scale_slot_mapping


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("block_size", [16, 128, 256])
def test_padded_scale_pages_preserve_unselected_blocks(dtype, block_size):
    width = scale_padding_width(block_size, dtype)
    cache = torch.full((7, block_size, 1, width), -7, dtype=dtype, device="cpu")
    slots = torch.tensor([3 * block_size + 5, block_size - 1, 5 * block_size], device="cpu")
    offsets = scale_slot_mapping(cache, slots)
    cache.view(-1)[offsets] = torch.tensor([1, 2, 3], dtype=dtype, device="cpu")
    view = scale_cache_for_indexer(cache)
    assert view.data_ptr() == cache.data_ptr()
    assert cache.stride(0) * cache.element_size() % 4096 == 0
    assert view[3, 5, 0] == 1
    assert view[0, block_size - 1, 0] == 2
    assert view[5, 0, 0] == 3
    assert torch.all(view[1] == -7)
    assert torch.all(view[:, block_size:] == -7)
    assert scale_slot_mapping(cache, torch.tensor([-1], device="cpu")).item() == -1


def test_unpadded_scales_keep_slot_identity():
    cache = torch.empty((3, 128, 1, 1), device="cpu")
    slots = torch.tensor([1, 128, -1], device="cpu")
    assert scale_slot_mapping(cache, slots) is slots
