"""Pin the reference implementation against the real scripts and the goldens.

``tests/helpers/legacy_ridge.py`` re-implements the direct brain -> feature
Ridge decoder in plain numpy + scikit-learn.  These tests are what make it
trustworthy: they show it reproduces what the actual scripts produce, so it can
later be used as the yardstick for the factorized implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import GOLDEN_DIR
from tests.helpers import legacy_ridge, pipeline

import os

ALPHA = 100
CHUNK_AXIS = 1


def reference_prediction(dataset, layer, subject, roi, alpha=ALPHA,
                         chunk_axis=CHUNK_AXIS):
    """Train and predict with the reference implementation."""
    brain = dataset.brain(subject, roi, 'train')
    brain_labels = dataset.labels(subject, 'train')
    feat_labels = list(np.unique(brain_labels))
    feat = dataset.feature_matrix(layer, feat_labels)

    trained = legacy_ridge.legacy_train(
        brain, brain_labels, feat, feat_labels, alpha=alpha,
        chunk_axis=chunk_axis)

    test_brain, test_labels = legacy_ridge.average_test_brain(
        dataset.brain(subject, roi, 'test'), dataset.labels(subject, 'test'))
    return trained, legacy_ridge.legacy_predict(trained, test_brain), test_labels


@pytest.fixture
def pipeline_outputs(dataset, decoder_dir, decoded_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS)
    return decoder_dir, decoded_dir


def test_reference_matches_scripts(dataset, pipeline_outputs):
    """The reference reproduces the shipped scripts, chunked and unchunked."""
    decoder_dir, decoded_dir = pipeline_outputs

    for layer in dataset.layers:
        for subject in sorted(dataset.train_fmri):
            for roi in sorted(dataset.rois):
                trained, expected, test_labels = reference_prediction(
                    dataset, layer, subject, roi)

                actual = pipeline.read_decoded_features(
                    decoded_dir, layer, subject, roi, test_labels)
                np.testing.assert_allclose(
                    actual, expected.astype(np.float32),
                    rtol=1e-5, atol=1e-6,
                    err_msg='prediction mismatch for %s/%s/%s'
                            % (layer, subject, roi))

                saved = pipeline.read_norm_params(decoder_dir, layer, subject,
                                                  roi)
                for key in pipeline.NORM_KEYS:
                    np.testing.assert_allclose(
                        saved[key], trained[key].astype(np.float32),
                        rtol=1e-6, atol=1e-7,
                        err_msg='%s mismatch for %s/%s/%s'
                                % (key, layer, subject, roi))


def test_reference_matches_golden(dataset):
    """The reference reproduces the committed pre-factorization fixtures."""
    golden = np.load(os.path.join(GOLDEN_DIR, 'sklearn_ridge_pipeline.npz'))
    assert int(golden['_alpha']) == ALPHA
    assert int(golden['_chunk_axis']) == CHUNK_AXIS

    for layer in [str(x) for x in golden['_layers']]:
        for subject in [str(x) for x in golden['_subjects']]:
            for roi in [str(x) for x in golden['_rois']]:
                trained, expected, test_labels = reference_prediction(
                    dataset, layer, subject, roi)
                assert test_labels == [str(x) for x in golden['_test_labels']]

                prefix = '%s|%s|%s|' % (layer, subject, roi)
                np.testing.assert_allclose(
                    golden[prefix + 'pred'], expected.astype(np.float32),
                    rtol=1e-5, atol=1e-6)
                for key in pipeline.NORM_KEYS:
                    np.testing.assert_allclose(
                        golden[prefix + key], trained[key].astype(np.float32),
                        rtol=1e-6, atol=1e-7)
