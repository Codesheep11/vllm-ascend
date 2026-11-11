# SPDX-License-Identifier: Apache-2.0
"""Aligned KV views must retain ownership of both registration boundary pages."""
from types import SimpleNamespace

import pytest
import torch

# Match production worker import ordering for device/operator registration.
import vllm_ascend.worker.worker  # noqa: F401
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


@pytest.mark.parametrize("numel", [1, 2097151, 2097152, 2097153, 3149824])
def test_transfer_cache_owns_rounded_registration_tail(numel: int) -> None:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.vllm_config = SimpleNamespace(kv_transfer_config=object())
    alignment = 2 * 1024 * 1024
    tensor = runner._allocate_int8_cache_tensor(numel, alignment)
    storage = tensor.untyped_storage()
    registered_bytes = (numel + alignment - 1) // alignment * alignment
    assert tensor.numel() == numel
    assert tensor.data_ptr() % alignment == 0
    assert tensor.data_ptr() + registered_bytes <= storage.data_ptr() + storage.nbytes()
    assert torch.count_nonzero(tensor) == 0
