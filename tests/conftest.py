"""Shared fixtures for the Phase 1 (M1.1) tokenizer contract tests.

The session-scoped ``trained_tokenizer_path`` fixture trains the real 16k BPE
once per test session on the committed bootstrap corpus
(``tests/fixtures/dialect_corpus.txt``) — deterministic, hermetic
(``BAYAN_OFFLINE=1`` safe), and fast enough at bootstrap scale.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer

from bayan_slm_engine.tokenizer.bpe_trainer import BPETrainer


@pytest.fixture(scope="session")
def dialect_corpus_path() -> Path:
    """Path to the committed M1.1 bootstrap dialect corpus."""
    return Path(__file__).parent / "fixtures" / "dialect_corpus.txt"


@pytest.fixture(scope="session")
def trained_tokenizer_path(
    tmp_path_factory: pytest.TempPathFactory, dialect_corpus_path: Path
) -> Path:
    """Train the real BPE once per session; return the persisted tokenizer.json path."""
    out = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    BPETrainer().train_from_path(dialect_corpus_path, out)
    return out


@pytest.fixture(scope="session")
def trained_tokenizer(trained_tokenizer_path: Path) -> Tokenizer:
    """The session-trained tokenizer object."""
    return Tokenizer.from_file(str(trained_tokenizer_path))
