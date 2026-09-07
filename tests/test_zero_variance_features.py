"""Feature units with zero variance across the training stimuli.

``ModelTraining`` z-scores the target as ``(Y - y_mean) / y_norm`` and then
cleans up only infinities (``Y[np.isinf(Y)] = 0``).  A unit that is constant
across the M unique training stimuli has ``y_norm == 0`` and ``Y - y_mean == 0``,
so the division yields ``NaN``, which survives the clean-up and scikit-learn
rejects.

That is the current, shipped behavior.  It is recorded here so that a change to
the decoder implementation cannot silently alter it; making such units decodable
is a separate bug fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import legacy_ridge, pipeline


def test_fixture_really_has_a_constant_unit(dataset_with_constant_unit):
    dataset = dataset_with_constant_unit
    for layer in dataset.layers:
        feat = dataset.features[layer].reshape(
            dataset.features[layer].shape[0], -1)
        assert np.std(feat[:, 0], ddof=1) == 0.0


def test_training_rejects_constant_feature_units(dataset_with_constant_unit,
                                                 decoder_dir):
    with pytest.raises(ValueError, match='NaN'):
        pipeline.run_training(dataset_with_constant_unit, decoder_dir,
                              alpha=100, chunk_axis=1, layers=['fc_like'])


def test_reference_implementation_rejects_them_too(dataset_with_constant_unit):
    dataset = dataset_with_constant_unit
    subject, roi, layer = 'sub-01', 'VC', 'fc_like'
    brain_labels = dataset.labels(subject, 'train')
    feat_labels = list(np.unique(brain_labels))

    with pytest.raises(ValueError, match='NaN'):
        legacy_ridge.legacy_train(
            dataset.brain(subject, roi, 'train'), brain_labels,
            dataset.feature_matrix(layer, feat_labels), feat_labels,
            alpha=100, chunk_axis=1)
