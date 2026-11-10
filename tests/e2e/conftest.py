"""Fixtures E2E: una única ejecución real compartida por toda la sesión de tests."""

from __future__ import annotations

import asyncio

import pytest

from src.config import AUDIT_DIR, PROJECT_ROOT
from src.utils.audit import RunAuditor
from tests.e2e.harness import (
    E2ERun,
    MatryoshkaAblationRun,
    execute_e2e_run,
    execute_matryoshka_ablation_run,
    execute_python_e2e_run,
)


@pytest.fixture(scope="session")
def e2e_run() -> E2ERun:
    """Indexa y evalúa animals.mp4 una sola vez en la sesión de tests."""
    with pytest.MonkeyPatch.context() as patch:
        # El pipeline guarda los fotogramas en una ruta relativa a data/processed.
        patch.chdir(PROJECT_ROOT)
        with RunAuditor(AUDIT_DIR) as auditor:
            return asyncio.run(execute_e2e_run(auditor))


@pytest.fixture(scope="session")
def python_e2e_run() -> E2ERun:
    """Indexa y evalúa python_tutorial.mp4 una sola vez en la sesión de tests."""
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(PROJECT_ROOT)
        with RunAuditor(AUDIT_DIR) as auditor:
            return asyncio.run(execute_python_e2e_run(auditor))


@pytest.fixture(scope="session")
def matryoshka_e2e_run() -> MatryoshkaAblationRun:
    """Indexa ambos vídeos (82 chunks c/u) y evalúa MRL (3072, 1536, 768) una sola vez."""
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(PROJECT_ROOT)
        with RunAuditor(AUDIT_DIR) as auditor:
            return asyncio.run(execute_matryoshka_ablation_run(auditor))


