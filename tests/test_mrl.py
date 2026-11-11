"""Pruebas del recorte Matryoshka con normalización L2."""

import numpy as np
import pytest

from src.embeddings.mrl import truncate_and_normalize_mrl


@pytest.mark.parametrize("target_dim", [3072, 1536, 768])
def test_truncate_and_normalize_mrl_dimensions_and_unit_norm(target_dim: int) -> None:
    rng = np.random.default_rng(42)
    raw = rng.normal(0.0, 1.0, size=3072).tolist()
    result = truncate_and_normalize_mrl(raw, target_dim)
    assert len(result) == target_dim
    assert float(np.linalg.norm(result)) == pytest.approx(1.0, rel=1e-6)
    expected_prefix = np.asarray(raw[:target_dim], dtype=np.float64)
    expected_norm = expected_prefix / np.linalg.norm(expected_prefix)
    assert np.allclose(result, expected_norm, atol=1e-6)


def test_truncate_and_normalize_mrl_validates_bounds_and_zero_vector() -> None:
    with pytest.raises(ValueError):
        truncate_and_normalize_mrl([1.0, 2.0, 3.0], 0)
    with pytest.raises(ValueError):
        truncate_and_normalize_mrl([1.0, 2.0, 3.0], 4)

    zeros = truncate_and_normalize_mrl([0.0, 0.0, 0.0, 0.0], 2)
    assert zeros == [0.0, 0.0]
