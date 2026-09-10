"""What the training features cost, and who reads them.

The factorized fit maps brain activity onto the training *stimulus basis*, so it
needs the trial labels and no feature values: **training never opens a feature
file**.  Prediction does, and there the invariant is that **no more than one
feature array is ever alive at a time**: for VGG19 ``conv1_1`` with 1200
training stimuli a single ``F`` is already ~15 GB.  Counting what the loader
holds is not enough -- it caches one layer by construction, and says nothing
about references the *caller* still keeps -- so these tests track the arrays
with ``weakref`` and count the ones that are actually still alive, and watch the
disk reads at bdpy's own loading function.
"""

from __future__ import annotations

import gc
import os
import weakref

import pytest

import bdpy.dataform.features as bdpy_features
from ridge_factorization import TrainingFeatureLoader
from tests.helpers import pipeline, synthetic

ALPHA = 100
CHUNK_AXIS = 1


class FeatureArrayTracker(object):
    """Track the arrays the loader hands out, and the reads behind them."""

    def __init__(self):
        self.refs = []
        self.max_alive = 0
        self.layers = []         # every request, cache hits included
        self.loaded_layers = []  # layers actually read from disk

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
    original_get = TrainingFeatureLoader.get
    original_load = bdpy_features._load_array_with_key

    def spy(self, layer, labels):
        features = original_get(self, layer, labels)
        tracker.observe(layer, features)
        return features

    def counted_load(key, path):
        # <features>/<layer>/<label>.mat -- a read that reached the disk.
        layer = os.path.basename(os.path.dirname(path))
        if layer not in tracker.loaded_layers:
            tracker.loaded_layers.append(layer)
        return original_load(key, path)

    monkeypatch.setattr(TrainingFeatureLoader, 'get', spy)
    monkeypatch.setattr(bdpy_features, '_load_array_with_key', counted_load)
    return tracker


@pytest.fixture
def no_feature_reads(monkeypatch):
    """Make any attempt to read a feature file fail the test."""
    def forbidden(self, layer, labels):
        raise AssertionError('the training features were read for layer %r'
                             % layer)

    monkeypatch.setattr(TrainingFeatureLoader, 'get', forbidden)


def test_loader_holds_a_single_layer(dataset):
    """One layer resident: the previous one is dropped before the next load."""
    loader = TrainingFeatureLoader([dataset.train_features_dir])
    labels = dataset.unique_train_labels

    first = loader.get('fc_like', labels)
    assert loader.get('fc_like', labels) is first  # served from the cache

    ref = weakref.ref(first)
    del first
    loader.get('conv_like', labels)
    gc.collect()
    assert ref() is None

    loader.release()


def test_training_does_not_open_a_feature_file(dataset, decoder_dir,
                                               no_feature_reads):
    """The point of the factorized fit: it needs the labels, not the features.

    Reading them was worth ~3 TB over a DeepRecon run (5 subjects x 9 ROIs x
    9 layers) purely to recompute identical per-column statistics.
    """
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    assert os.path.isfile(os.path.join(
        pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC'),
        'model.pkl.gz'))


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
    assert sorted(tracker.loaded_layers) == sorted(dataset.layers)
