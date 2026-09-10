"""What the training features cost, and who reads them.

The factorized fit maps brain activity onto the training *stimulus basis*, so it
needs the trial labels and no feature values: **training never opens a feature
file**.  Prediction does, and there the invariant is that **no more than one
feature array is ever alive at a time**: for VGG19 ``conv1_1`` with 1200
training stimuli a single ``F`` is already ~15 GB.  Asserting on
``TrainingFeatureLoader.cached_layers`` alone cannot check that -- it is a
one-element list by construction, and it says nothing about references the
*caller* still holds -- so these tests track the arrays with ``weakref`` and
count the ones that are actually still alive.
"""

from __future__ import annotations

import gc
import os
import weakref

import numpy as np
import pytest

from ridge_factorization import TrainingFeatureLoader
from tests.helpers import pipeline, synthetic

ALPHA = 100
CHUNK_AXIS = 1


class FeatureArrayTracker(object):
    """Track how many arrays returned by the loader are still reachable."""

    def __init__(self):
        self.refs = []
        self.alive_on_entry = []
        self.max_alive = 0
        self.layers = []         # every request, cache hits included
        self.loaded_layers = []  # only the requests that hit the disk

    def live_count(self) -> int:
        gc.collect()
        return len({id(obj) for obj in (ref() for ref in self.refs)
                    if obj is not None})

    def observe(self, layer, features):
        self.layers.append(layer)
        self.refs.append(weakref.ref(features))
        self.max_alive = max(self.max_alive, self.live_count())


@pytest.fixture
def tracker(monkeypatch):
    tracker = FeatureArrayTracker()
    original = TrainingFeatureLoader.get

    def spy(self, layer, labels):
        tracker.alive_on_entry.append(tracker.live_count())
        loads_before = self.load_count
        features = original(self, layer, labels)
        if self.load_count != loads_before:
            tracker.loaded_layers.append(layer)
        tracker.observe(layer, features)
        return features

    monkeypatch.setattr(TrainingFeatureLoader, 'get', spy)
    return tracker


@pytest.fixture
def no_feature_reads(monkeypatch):
    """Make any attempt to read a feature file fail the test."""
    def forbidden(self, layer, labels):
        raise AssertionError('the training features were read for layer %r'
                             % layer)

    monkeypatch.setattr(TrainingFeatureLoader, 'get', forbidden)


def test_loader_holds_a_single_layer(dataset):
    loader = TrainingFeatureLoader([dataset.train_features_dir])
    labels = dataset.unique_train_labels

    first = loader.get('fc_like', labels)
    assert loader.cached_layers == ['fc_like']

    # Same request: served from the cache, no extra read.
    assert loader.get('fc_like', labels) is first
    assert loader.load_count == 1

    loader.get('conv_like', labels)
    assert loader.cached_layers == ['conv_like']
    assert loader.load_count == 2

    loader.release()
    assert loader.cached_layers == []


def test_loader_drops_its_reference_before_loading_the_next_layer(dataset):
    loader = TrainingFeatureLoader([dataset.train_features_dir])
    labels = dataset.unique_train_labels

    ref = weakref.ref(loader.get('fc_like', labels))
    loader.get('conv_like', labels)
    gc.collect()

    assert ref() is None


def test_loader_returns_rows_in_the_requested_order(dataset):
    loader = TrainingFeatureLoader([dataset.train_features_dir])
    labels = list(reversed(dataset.unique_train_labels))

    features = loader.get('conv_like', labels)

    np.testing.assert_array_equal(features,
                                  dataset.feature_matrix('conv_like', labels))


def test_training_does_not_open_a_feature_file(dataset, decoder_dir,
                                               no_feature_reads):
    """The point of the factorized fit: it needs the labels, not the features.

    Reading them was worth ~3 TB over a DeepRecon run (5 subjects x 9 ROIs x
    9 layers) purely to recompute identical per-column statistics.
    """
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    for layer in dataset.layers:
        assert os.path.isfile(os.path.join(
            pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC'),
            'model.pkl.gz'))


def test_training_reads_nothing_by_default_and_one_layer_at_a_time(
        dataset, decoder_dir, tracker):
    """Same statement measured through the loader rather than an exception."""
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    assert tracker.layers == []
    assert tracker.loaded_layers == []


def test_prediction_never_holds_two_layers_of_features(dataset, decoder_dir,
                                                       decoded_dir, tracker):
    synthetic.add_layer(dataset, 'extra_like', (5,))

    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS)

    assert tracker.max_alive == 1
    assert tracker.live_count() == 0


def test_prediction_reads_each_layer_once(dataset, decoder_dir, decoded_dir,
                                          tracker):
    """One disk read per layer, not one per (layer, subject, ROI).

    Every (layer, subject, ROI) asks for the features twice on a first run --
    once to write the statistics sidecars its decoder directory needs, once to
    decode -- and the loader serves all of that from the one resident layer.
    """
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS)

    combinations = (len(dataset.layers) * len(dataset.test_fmri)
                    * len(dataset.rois))
    assert len(tracker.layers) == 2 * combinations
    assert tracker.loaded_layers == list(reversed(dataset.layers))
