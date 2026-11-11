"""Recorte Matryoshka (MRL) de embeddings con normalización L2."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["truncate_and_normalize_mrl"]


def truncate_and_normalize_mrl(
    vector: Sequence[float],
    target_dim: int,
) -> list[float]:
    """Recorta un vector a ``target_dim`` dimensiones Matryoshka y lo normaliza en norma L2."""
    if target_dim < 1 or target_dim > len(vector):
        raise ValueError(
            f"target_dim debe estar en [1, {len(vector)}], recibido: {target_dim}."
        )
    sliced = [float(v) for v in vector[:target_dim]]
    norm = float(np.linalg.norm(sliced))
    if norm > 0.0:
        return [v / norm for v in sliced]
    return sliced
