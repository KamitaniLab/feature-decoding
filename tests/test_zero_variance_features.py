"""Feature units with zero variance across the training stimuli.

The *direct* decoder z-scored the target as ``(Y - y_mean) / y_norm`` and then
cleaned up only infinities (``Y[np.isinf(Y)] = 0``).  A unit that is constant
across the M unique training stimuli has ``y_norm == 0`` and ``Y - y_mean == 0``,
so the division yielded ``NaN``, which survived the clean-up and scikit-learn
rejected it.  That behavior is recorded here against the reference
implementation, which is what the golden fixtures were produced with.

The factorized decoder does not reproduce it. It never normalizes the features
and decodes a constant unit correctly as ``coeff @ F``, and since training no
longer reads the features it cannot detect the condition at all. The divergence
is deliberate and is called out in the PR.
"""

from __future__ import annotations

import numpy as np
import pytest

import ridge_factorization

from tests.helpers import legacy_ridge, pipeline


def test_fixture_really_has_a_constant_unit(dataset_with_constant_unit):
    dataset = dataset_with_constant_unit
    for layer in dataset.layers:
        feat = dataset.features[layer].reshape(
            dataset.features[layer].shape[0], -1)
        assert np.std(feat[:, 0], ddof=1) == 0.0


def test_factorized_training_accepts_constant_feature_units(
        dataset_with_constant_unit, decoder_dir):
    """The deliberate divergence: the factorized path no longer rejects them."""
    pipeline.run_training(dataset_with_constant_unit, decoder_dir,
                          alpha=100)

    assert ridge_factorization.is_factorized_model_dir(
        pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC'))


def test_factorized_prediction_decodes_constant_units_exactly(
        dataset_with_constant_unit, decoder_dir, decoded_dir):
    """A constant unit decodes to its constant value rather than failing."""
    dataset = dataset_with_constant_unit
    pipeline.run_training(dataset, decoder_dir, alpha=100, subjects=['sub-01'],
                          rois={'VC': 'ROI_VC = 1'})
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir, chunk_axis=1,
                            layers=['fc_like'], subjects=['sub-01'],
                            rois={'VC': 'ROI_VC = 1'})

    predicted = pipeline.read_decoded_features(
        decoded_dir, 'fc_like', 'sub-01', 'VC', dataset.unique_test_labels)
    constant = dataset.features['fc_like'].reshape(
        dataset.features['fc_like'].shape[0], -1)[0, 0]
    np.testing.assert_allclose(predicted[:, 0], constant, rtol=1e-5)


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
