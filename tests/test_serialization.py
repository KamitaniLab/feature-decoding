"""Round trip: train -> serialize -> load -> predict."""

from __future__ import annotations

import os
import pickle

import numpy as np

import ridge_factorization
from ridge_factorization import combine_features, load_train_labels
from tests.helpers import legacy_ridge, pipeline

ALPHA = 100
CHUNK_AXIS = 1


def test_reloaded_model_reproduces_the_saved_prediction(dataset, decoder_dir,
                                                        decoded_dir):
    """Recomputing the decode by hand from the artifact gives the same output."""
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS)

    for layer in dataset.layers:
        subject, roi = 'sub-01', 'VC'
        directory = pipeline.brain_model_dir(decoder_dir, subject, roi)

        with open(ridge_factorization.model_file_path(directory), 'rb') as f:
            model = pickle.load(f)['model']
        train_labels = load_train_labels(directory)
        params = pipeline.read_norm_params(decoder_dir, subject, roi,
                                           keys=pipeline.BRAIN_NORM_KEYS)

        test_brain, test_labels = legacy_ridge.average_test_brain(
            dataset.brain(subject, roi, 'test'),
            dataset.labels(subject, 'test'))
        normalized = ((test_brain - params['x_mean']) / params['x_norm'])
        coefficients = model.predict(normalized.astype(np.float32))
        features = dataset.feature_matrix(layer, train_labels)
        expected = combine_features(coefficients, features.astype(np.float32),
                                    chunk_axis=CHUNK_AXIS)

        actual = pipeline.read_decoded_features(decoded_dir, layer, subject,
                                                roi, test_labels)
        np.testing.assert_allclose(actual, expected.astype(np.float32),
                                   rtol=1e-6, atol=1e-7)


def test_prediction_is_reproducible_across_runs(dataset, decoder_dir,
                                                tmp_path):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    outputs = []
    for i in range(2):
        decoded_dir = str(tmp_path / ('decoded_%d' % i))
        pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                                chunk_axis=CHUNK_AXIS, layers=['fc_like'],
                                analysis_name='predict_%d' % i)
        outputs.append(pipeline.read_decoded_features(
            decoded_dir, 'fc_like', 'sub-01', 'VC',
            dataset.unique_test_labels))

    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_artifact_file_set(dataset, decoder_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    directory = pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC')
    assert sorted(os.listdir(directory)) == [
        'factorized.yaml',
        'info.yaml',
        'model.pkl.gz',
        'train_labels.json',
        'x_mean.mat',
        'x_norm.mat',
    ]
