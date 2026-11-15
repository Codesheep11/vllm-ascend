# SPDX-License-Identifier: Apache-2.0
"""Native C8 indexer must read logical scales with the physical page stride."""

import pytest
import torch
import torch_npu

from vllm_ascend.utils import AscendDeviceType, enable_custom_op, get_ascend_device_type
from vllm_ascend.worker.xds_layout import scale_cache_for_indexer, scale_padding_width, scale_slot_mapping


@pytest.mark.parametrize("lengths", [[127, 128, 129], [2047, 2048, 2049], [3001, 3500, 4096]])
def test_xds_indexer_scale_padding(lengths):
    if get_ascend_device_type() == AscendDeviceType.A5:
        pytest.skip("A5 uses a different indexer kernel")
    assert enable_custom_op(), "Rebuild the custom indexer operator before this test"
    torch.npu.set_device(0)
    torch.manual_seed(41)
    blocks = 96
    query = torch.randint(-30, 31, (3, 64, 128), dtype=torch.int8, device="cpu").npu()
    key = torch.randint(-30, 31, (blocks, 128, 1, 128), dtype=torch.int8, device="cpu").npu()
    weights = torch.rand((3, 64), dtype=torch.float16, device="cpu").npu()
    qs = (torch.rand((3, 64), dtype=torch.float16, device="cpu") * 0.1 + 0.01).npu()
    scales = (torch.rand((blocks, 128, 1), dtype=torch.float16, device="cpu") * 0.1 + 0.01).npu()
    padded = torch.full(
        (blocks, 128, 1, scale_padding_width(128, torch.float16)), 500, dtype=torch.float16, device="npu"
    )
    slots = torch.arange(blocks * 128, device="npu", dtype=torch.int64)
    torch_npu.npu_scatter_nd_update_(
        padded.view(-1, 1), scale_slot_mapping(padded, slots).view(-1, 1), scales.view(-1, 1)
    )
    bt = torch.randperm(blocks, device="cpu").reshape(3, 32).to(torch.int32).npu()
    args = dict(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=qs,
        actual_seq_lengths_query=torch.tensor([1, 2, 3], dtype=torch.int32).npu(),
        actual_seq_lengths_key=torch.tensor(lengths, dtype=torch.int32).npu(),
        block_table=bt,
        query_quant_mode=0,
        key_quant_mode=0,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=2048,
        sparse_mode=3,
    )
    dense = torch.ops._C_ascend.npu_lightning_indexer_quant(key_dequant_scale=scales, **args)
    aligned = torch.ops._C_ascend.npu_lightning_indexer_quant(key_dequant_scale=scale_cache_for_indexer(padded), **args)
    torch.npu.synchronize()
    assert torch.equal(dense.cpu(), aligned.cpu()), f"index mismatch: {lengths}"
    assert torch.all(padded.view(blocks, -1)[:, 128:] == 500).item()
