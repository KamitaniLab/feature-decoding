"""Pin the reference implementation against the real scripts.

``tests/helpers/legacy_ridge.py`` re-implements the direct brain -> feature
Ridge decoder in plain numpy + scikit-learn.  These tests are what make it
trustworthy: they show it reproduces what the actual scripts produce, so it can
later be used as the yardstick for the factorized implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers import legacy_ridge, pipeline

ALPHA = 100
CHUNK_AXIS = 1

# The baseline plus one departure from it each, so the regularization, the
# chunking and the trial averaging are all covered.  ``chunk_axis=2`` is
# deliberately absent: this test runs every layer, and a 2-D layer is stored as
# a single unchunked model, which ``ModelTest`` would then try to reassemble
# along an axis it does not have.  ``test_chunking.py`` covers axis 2 on the
# multidimensional layer, where it is meaningful.
CONFIGURATIONS = (
    pytest.param(ALPHA, CHUNK_AXIS, True, id='baseline'),
    pytest.param(1, CHUNK_AXIS, True, id='alpha-1'),
    pytest.param(ALPHA, None, True, id='unchunked'),
    pytest.param(ALPHA, CHUNK_AXIS, False, id='single-trial'),
)


def reference_prediction(dataset, layer, subject, roi, alpha=ALPHA,
                         chunk_axis=CHUNK_AXIS, average_sample=True):
    """Train and predict with the reference implementation."""
    brain = dataset.brain(subject, roi, 'train')
    brain_labels = dataset.labels(subject, 'train')
    feat_labels = list(np.unique(brain_labels))
    feat = dataset.feature_matrix(layer, feat_labels)

    trained = legacy_ridge.legacy_train(
        brain, brain_labels, feat, feat_labels, alpha=alpha,
        chunk_axis=chunk_axis)

    test_brain = dataset.brain(subject, roi, 'test')
    trial_labels = dataset.labels(subject, 'test')
    if average_sample:
        test_brain, test_labels = legacy_ridge.average_test_brain(
            test_brain, trial_labels)
    else:
        test_labels = legacy_ridge.single_trial_labels(trial_labels)

    return trained, legacy_ridge.legacy_predict(trained, test_brain), test_labels


@pytest.mark.parametrize('alpha, chunk_axis, average_sample', CONFIGURATIONS)
def test_reference_matches_scripts(dataset, decoder_dir, decoded_dir, alpha,
                                   chunk_axis, average_sample):
    """The reference reproduces the shipped scripts, in every configuration."""
    pipeline.run_training(dataset, decoder_dir, alpha=alpha)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=chunk_axis,
                            average_sample=average_sample)

    for layer in dataset.layers:
        for subject in sorted(dataset.train_fmri):
            for roi in sorted(dataset.rois):
                trained, expected, test_labels = reference_prediction(
                    dataset, layer, subject, roi, alpha=alpha,
                    chunk_axis=chunk_axis, average_sample=average_sample)

                # The factorized path reaches the same numbers through a
                # different product, so float32 rounding differs; a small
                # alpha leaves larger coefficients and makes it visible.
                rtol, atol = (1e-5, 1e-6) if alpha >= 100 else (1e-4, 1e-5)
                actual = pipeline.read_decoded_features(
                    decoded_dir, layer, subject, roi, test_labels)
                np.testing.assert_allclose(
                    actual, expected.astype(np.float32), rtol=rtol, atol=atol,
                    err_msg='prediction mismatch for %s/%s/%s'
                            % (layer, subject, roi))

                saved = pipeline.read_norm_params(decoder_dir, subject, roi)
                saved.update(pipeline.read_feature_statistics(
                    decoder_dir, layer, subject, roi))
                for key in legacy_ridge.NORM_KEYS:
                    np.testing.assert_allclose(
                        saved[key], trained[key].astype(np.float32),
                        rtol=1e-6, atol=1e-7,
                        err_msg='%s mismatch for %s/%s/%s'
                                % (key, layer, subject, roi))
