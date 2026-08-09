"""Integration tests for the fine-tuning loop's cross-rank evaluation reduction."""

from __future__ import annotations

import math
import os
import socket

import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

from dr_model.training.distributed import (
    cleanup_distributed,
    detect_distributed_context,
    init_distributed,
)
from dr_model.training.finetune_loop import _evaluate


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _eval_worker(rank: int, world_size: int, port: int, seed: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    ctx = detect_distributed_context("cpu")
    init_distributed(ctx)
    try:
        gen = torch.Generator().manual_seed(seed + rank)
        images = torch.randn(8, 3, 384, 384, generator=gen)
        labels = torch.randint(0, 5, (8,), generator=gen)

        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 384 * 384, 5))
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(generator=gen)

        dl = DataLoader(TensorDataset(images, labels), batch_size=8)
        criterion = nn.CrossEntropyLoss()
        mean_loss, kappa = _evaluate(model, dl, criterion, torch.device("cpu"), world_size)

        model.eval()
        with torch.no_grad():
            loss = criterion(model(images), labels)
            preds = model(images).argmax(dim=1).tolist()
        local_labels = labels.tolist()

        loss_t = torch.tensor([loss.item()])
        cnt_t = torch.tensor([1.0])  # one batch of 8 per rank
        torch.distributed.all_reduce(loss_t, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(cnt_t, op=torch.distributed.ReduceOp.SUM)

        labels_by_rank: list[list[int]] = [[] for _ in range(world_size)]
        preds_by_rank: list[list[int]] = [[] for _ in range(world_size)]
        torch.distributed.all_gather_object(labels_by_rank, local_labels)
        torch.distributed.all_gather_object(preds_by_rank, preds)

        all_labels = [label for labels_ in labels_by_rank for label in labels_]
        all_preds = [pred for preds_ in preds_by_rank for pred in preds_]

        expected_loss = loss_t.item() / cnt_t.item()
        expected_kappa = float(cohen_kappa_score(all_labels, all_preds, weights="quadratic"))

        assert math.isclose(mean_loss, expected_loss)
        assert math.isclose(kappa, expected_kappa)
    finally:
        cleanup_distributed()


class TestEvaluateReduction:
    def test_reduces_across_ranks(self) -> None:
        world_size = 2

        torch.multiprocessing.start_processes(
            _eval_worker,
            args=(world_size, _free_port(), 1234),
            nprocs=world_size,
            join=True,
        )
