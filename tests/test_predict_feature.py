"""Behavior of ``predict_feature.featdec_predict``.

Pins the decoded-feature output contract: one ``<label>.mat`` per predicted
sample under ``<output>/<layer>/<subject>/<roi>/``, key ``feat``, shape
``(1, *feature_shape)``, ``float32`` -- the layout
``bdpy.dataform.DecodedFeatures`` (and therefore ``evaluation.py``) expects.
"""

from __future__ import annotations

import os

import numpy as np

from tests.helpers import legacy_ridge, pipeline

ALPHA = 100
CHUNK_AXIS = 1


def test_output_layout_and_dtype(dataset, decoder_dir, decoded_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS)

    from bdpy.dataform import load_array

    for layer in dataset.layers:
        feature_shape = dataset.layer_shapes[layer]
        for subject in dataset.test_fmri:
            for roi in dataset.rois:
                directory = os.path.join(decoded_dir, layer, subject, roi)
                names = pipeline.decoded_feature_labels(decoded_dir, layer,
                                                        subject, roi)
                assert names == dataset.unique_test_labels
                for label in names:
                    array = load_array(
                        os.path.join(directory, '%s.mat' % label), key='feat')
                    assert array.shape == (1,) + feature_shape
                    assert array.dtype == np.float32


def test_average_sample_averages_repetitions(dataset, decoder_dir,
                                             decoded_dir):
    """With ``average_sample`` on, one file per unique stimulus is written."""
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS, average_sample=True)

    names = pipeline.decoded_feature_labels(decoded_dir, dataset.layers[0],
                                            'sub-01', 'VC')
    assert names == dataset.unique_test_labels
    assert len(names) < len(dataset.labels('sub-01', 'test'))


def test_single_trial_output_names(dataset, decoder_dir, decoded_dir):
    """With ``average_sample`` off, files are ``sample%06d-<label>.mat``."""
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS, average_sample=False)

    trial_labels = dataset.labels('sub-01', 'test')
    expected = sorted(legacy_ridge.single_trial_labels(trial_labels))
    names = pipeline.decoded_feature_labels(decoded_dir, dataset.layers[0],
                                            'sub-01', 'VC')
    assert names == expected


def test_excluded_labels_are_dropped_when_averaging(dataset, decoder_dir,
                                                    decoded_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS)
    excluded = [dataset.unique_test_labels[0]]
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS, average_sample=True,
                            excluded_labels=excluded)

    names = pipeline.decoded_feature_labels(decoded_dir, dataset.layers[0],
                                            'sub-01', 'VC')
    assert names == dataset.unique_test_labels[1:]


def test_existing_output_directory_is_skipped(dataset, decoder_dir,
                                              decoded_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS)

    layer, subject, roi = dataset.layers[0], 'sub-01', 'VC'
    target = os.path.join(decoded_dir, layer, subject, roi,
                          '%s.mat' % dataset.unique_test_labels[0])
    mtime = os.stat(target).st_mtime_ns

    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS,
                            analysis_name='test_prediction_rerun')

    assert os.stat(target).st_mtime_ns == mtime


def test_prediction_uses_the_decoders_brain_normalization(dataset, decoder_dir,
                                                          decoded_dir):
    """Test brain data is z-scored with the *training* mean/SD, not its own."""
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS)

    layer, subject, roi = 'fc_like', 'sub-01', 'VC'
    brain_labels = dataset.labels(subject, 'train')
    feat_labels = list(np.unique(brain_labels))
    trained = legacy_ridge.legacy_train(
        dataset.brain(subject, roi, 'train'), brain_labels,
        dataset.feature_matrix(layer, feat_labels), feat_labels,
        alpha=ALPHA, chunk_axis=CHUNK_AXIS)

    test_brain, test_labels = legacy_ridge.average_test_brain(
        dataset.brain(subject, roi, 'test'), dataset.labels(subject, 'test'))
    expected = legacy_ridge.legacy_predict(trained, test_brain)
    actual = pipeline.read_decoded_features(decoded_dir, layer, subject, roi,
                                            test_labels)
    np.testing.assert_allclose(actual, expected.astype(np.float32),
                               rtol=1e-5, atol=1e-6)

    # Z-scoring the test data with its own statistics would give something else.
    self_scaled = ((test_brain - np.mean(test_brain, axis=0)[np.newaxis, :])
                   / np.std(test_brain, axis=0, ddof=1)[np.newaxis, :])
    other = legacy_ridge.legacy_predict(
        {**trained, 'x_mean': np.zeros_like(trained['x_mean']),
         'x_norm': np.ones_like(trained['x_norm'])}, self_scaled)
    assert not np.allclose(actual, other, rtol=1e-3)
