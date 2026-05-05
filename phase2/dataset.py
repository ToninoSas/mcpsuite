"""
PyTorch Dataset over memory-mapped activation files produced by
phase1/pipeline.py --capture_activations.

Layout on disk:
    activations/{split}/
        X.npy        float16  (N, 32, 4096)
        y.npy        int8     (N,)
        meta.jsonl   id, category, hallucination_type per row
        shape.json   {"X_shape": [N, 32, 4096], ...}

Usage:
    ds = ActivationDataset("../phase1/outputs/activations/train", layer=15)
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    # iterate all layers:
    for layer_idx in range(32):
        ds = ActivationDataset(split_dir, layer=layer_idx)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ActivationDataset(Dataset):
    """
    Memory-mapped dataset for a single layer's activations.

    Args:
        split_dir:  path to activations/{split}/  containing X.npy, y.npy, meta.jsonl
        layer:      which transformer layer to expose (0-31)
        filter_inference_errors:
                    if True (default), drop rows where hallucination_type == INFERENCE_ERROR
    """

    def __init__(
        self,
        split_dir: str | Path,
        layer: int,
        filter_inference_errors: bool = True,
    ) -> None:
        split_dir = Path(split_dir)

        X_path = split_dir / "X.npy"
        y_path = split_dir / "y.npy"
        meta_path = split_dir / "meta.jsonl"

        for p in (X_path, y_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(p)

        meta = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]

        # memory-map so the full array never lands in RAM
        X_full = np.load(X_path, mmap_mode="r")   # (N, 32, 4096) float16
        y_full = np.load(y_path, mmap_mode="r")    # (N,)          int8

        n_layers = X_full.shape[1]
        if not (0 <= layer < n_layers):
            raise ValueError(f"layer={layer} out of range [0, {n_layers})")

        self.layer = layer

        if filter_inference_errors:
            valid_idx = [
                i for i, m in enumerate(meta)
                if m.get("hallucination_type") != "INFERENCE_ERROR"
            ]
        else:
            valid_idx = list(range(len(meta)))

        # copy the slice we need into RAM (only one layer, not all 32)
        # shape: (N_valid, 4096)  still small: 1400 × 4096 × 2 bytes ≈ 11 MB
        self.X = torch.from_numpy(
            X_full[valid_idx, layer, :].astype(np.float32)
        )
        self.y = torch.from_numpy(
            y_full[valid_idx].astype(np.float32)
        )
        self.meta = [meta[i] for i in valid_idx]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @property
    def pos_weight(self) -> torch.Tensor:
        """pos_weight = n_neg / n_pos  for BCELoss."""
        n_pos = self.y.sum().item()
        n_neg = len(self.y) - n_pos
        if n_pos == 0:
            return torch.tensor(1.0)
        return torch.tensor(n_neg / n_pos)

    @property
    def class_counts(self) -> dict[str, int]:
        n_pos = int(self.y.sum().item())
        return {"neg": len(self.y) - n_pos, "pos": n_pos}
