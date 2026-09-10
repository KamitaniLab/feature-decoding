"""Shared fixtures for the feature-decoding tests."""

from __future__ import annotations

import os

import pytest

from tests.helpers import synthetic

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'data', 'golden')


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    """Run every test in its own directory.

    Both scripts create ``./tmp/<analysis_name>.db`` (the DistComp sqlite lock
    database) relative to the current working directory.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def dataset(tmp_path):
    """A small synthetic dataset with repeated stimuli."""
    return synthetic.make_dataset(str(tmp_path / 'data'), seed=0)


@pytest.fixture
def dataset_with_constant_unit(tmp_path):
    """Synthetic dataset where one feature unit has zero training variance."""
    return synthetic.make_dataset(str(tmp_path / 'data'), seed=0,
                                  constant_feature_unit=True)


@pytest.fixture
def decoder_dir(tmp_path):
    return str(tmp_path / 'feature_decoders')


@pytest.fixture
def decoded_dir(tmp_path):
    return str(tmp_path / 'decoded_features')
