# SPDX-License-Identifier: Apache-2.0
import unittest
from unittest.mock import patch


class TestNPUWorkerShutdown(unittest.TestCase):
    def test_kv_connector_closes_before_worker_memory_teardown(self) -> None:
        from vllm_ascend.worker.worker import NPUWorker

        events = []
        worker = NPUWorker.__new__(NPUWorker)
        with (
            patch("vllm_ascend.worker.worker.ensure_kv_transfer_shutdown", side_effect=lambda: events.append("kv")),
            patch("vllm.v1.worker.worker_base.WorkerBase.shutdown", side_effect=lambda: events.append("worker")),
        ):
            worker.shutdown()
        self.assertEqual(events, ["kv", "worker"])

    def test_interpreter_shutdown_allows_missing_kv_function(self) -> None:
        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        with (
            patch("vllm_ascend.worker.worker.ensure_kv_transfer_shutdown", None),
            patch("vllm.v1.worker.worker_base.WorkerBase.shutdown") as parent_shutdown,
        ):
            worker.shutdown()
        parent_shutdown.assert_called_once()

    def test_failed_kv_drain_prevents_memory_teardown(self) -> None:
        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        with (
            patch("vllm_ascend.worker.worker.ensure_kv_transfer_shutdown", side_effect=RuntimeError("DMA pending")),
            patch("vllm.v1.worker.worker_base.WorkerBase.shutdown") as parent_shutdown,
            self.assertRaisesRegex(RuntimeError, "DMA pending"),
        ):
            worker.shutdown()
        parent_shutdown.assert_not_called()
