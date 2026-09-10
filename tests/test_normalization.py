"""Normalization semantics of the factorized decoder.

Brain normalization is unchanged.  Feature normalization is where the
factorization removes work: the round trip ``(Y - y_mean) / y_norm`` at training
followed by ``* y_norm + y_mean`` at prediction cancels exactly (see
``ridge_factorization``), so the factorized prediction combines the raw training
features.  ``y_mean.mat`` / ``y_norm.mat`` still exist in the decoder's
per-layer directory because ``evaluation.py`` normalizes the pattern
correlation with them, but nothing in the decoding needs them, so
``predict_feature.py`` writes them as compatibility sidecars (see
``tests/test_lazy_feature_statistics.py``).
"""

from __future__ import annotations

import os

import numpy as np

from bdpy.dataform import load_array, save_array
from ridge_factorization import normalize_brain_for_training
from tests.helpers import pipeline

ALPHA = 100
CHUNK_AXIS = 1


def test_brain_normalization_zeroes_infinities():
    """Matches ``ModelTraining.run()``: infinities become zero, NaNs do not."""
    brain = np.array([[1.0, 2.0, 3.0],
                      [1.0, 4.0, 3.0]])
    mean = np.array([[0.0, 3.0, 3.0]])
    std = np.array([[0.0, 1.0, 0.0]])

    normalized = normalize_brain_for_training(brain, mean, std)

    # Column 0: nonzero deviation over zero SD -> inf -> replaced by 0.
    np.testing.assert_array_equal(normalized[:, 0], [0.0, 0.0])
    # Column 1: ordinary z-score.
    np.testing.assert_allclose(normalized[:, 1], [-1.0, 1.0])
    # Column 2: 0/0 -> NaN, deliberately left as is.
    assert np.all(np.isnan(normalized[:, 2]))


def test_feature_statistics_are_materialized_by_prediction(
        dataset, decoder_dir, decoded_dir):
    """Training cannot compute them -- it never reads the features.

    ``evaluation.py`` reads them from ``<layer>/<subject>/<roi>/model/``, so
    the definition has to stay exactly what the direct implementation wrote:
    per-unit statistics over the unique stimuli, with ``ddof=1``.
    """
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    for layer in dataset.layers:
        for subject in dataset.train_fmri:
            for roi in dataset.rois:
                assert not os.path.exists(os.path.join(
                    pipeline.layer_model_dir(decoder_dir, layer, subject, roi),
                    'y_mean.mat'))

    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS)

    for layer in dataset.layers:
        for subject in dataset.test_fmri:
            feat = dataset.feature_matrix(
                layer, list(np.unique(dataset.labels(subject, 'train'))))
            for roi in dataset.rois:
                params = pipeline.read_feature_statistics(decoder_dir, layer,
                                                          subject, roi)
                np.testing.assert_allclose(
                    params['y_mean'],
                    np.mean(feat, axis=0)[np.newaxis].astype(np.float32),
                    rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(
                    params['y_norm'],
                    np.std(feat, axis=0, ddof=1)[np.newaxis].astype(np.float32),
                    rtol=1e-5, atol=1e-6)


def test_prediction_does_not_depend_on_the_feature_statistics(dataset,
                                                              tmp_path):
    """Corrupting ``y_mean``/``y_norm`` must not change what is decoded.

    A direct demonstration that the feature normalization cancels: the
    factorized prediction never reads these files.  They are corrupted in the
    decoder and the features are decoded again into a fresh directory, so the
    second run really does see them.
    """
    decoder_dir = str(tmp_path / 'decoders')

    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)
    pipeline.run_prediction(dataset, decoder_dir, str(tmp_path / 'intact'),
                            chunk_axis=CHUNK_AXIS,
                            analysis_name='predict_intact')

    intact = {layer: pipeline.read_decoded_features(
        str(tmp_path / 'intact'), layer, 'sub-01', 'VC',
        dataset.unique_test_labels) for layer in dataset.layers}

    for layer in dataset.layers:
        for subject in dataset.test_fmri:
            for roi in dataset.rois:
                directory = pipeline.layer_model_dir(decoder_dir, layer,
                                                     subject, roi)
                for key in ('y_mean', 'y_norm'):
                    path = os.path.join(directory, '%s.mat' % key)
                    array = load_array(path, key=key)
                    save_array(path, array * 3.0 + 17.0, key=key,
                               dtype=np.float32, sparse=False)

    pipeline.run_prediction(dataset, decoder_dir, str(tmp_path / 'corrupted'),
                            chunk_axis=CHUNK_AXIS,
                            analysis_name='predict_corrupted')

    for layer in dataset.layers:
        np.testing.assert_array_equal(
            pipeline.read_decoded_features(str(tmp_path / 'corrupted'), layer,
                                           'sub-01', 'VC',
                                           dataset.unique_test_labels),
            intact[layer])
