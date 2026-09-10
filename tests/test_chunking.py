"""Chunking of the feature dimension.

Chunking used to be a *training* concern: bdpy fitted one Ridge per index along
``chunk_axis`` and flattened each chunk with ``order='F'``.  In the factorized
decoder the fitted model no longer touches the feature dimensions at all, so
chunking is purely a prediction-time memory device: the linear combination with
the training features is evaluated one index at a time along ``chunk_axis``.

Either way ``chunk_axis`` must not change the result, and each feature position
must depend only on its own training features.  The shipped config only enables
``fc6``/``fc7``/``fc8``, which are 2-D and never reach this path, so it is
pinned explicitly here.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from ridge_factorization import combine_features
from tests.helpers import pipeline

ALPHA = 100


def _model_files(decoder_dir, layer, subject='sub-01', roi='VC'):
    directory = pipeline.brain_model_dir(decoder_dir, subject, roi)
    return sorted(f for f in os.listdir(directory) if f.endswith('.pkl.gz'))


def test_one_model_per_layer_regardless_of_feature_dimensionality(dataset,
                                                                  decoder_dir):
    """No per-chunk model files: the stored model is the stimulus-basis Ridge."""
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    for layer in dataset.layers:
        assert _model_files(decoder_dir, layer) == ['model.pkl.gz']


def test_stored_model_targets_the_stimulus_basis(dataset, decoder_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    n_stimuli = len(np.unique(dataset.labels('sub-01', 'train')))
    for layer in dataset.layers:
        directory = pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC')
        with open(os.path.join(directory, 'model.pkl.gz'), 'rb') as f:
            payload = pickle.load(f)
        assert payload['y_shape'] == (n_stimuli,)
        n_voxels = dataset.brain('sub-01', 'VC').shape[1]
        assert payload['model'].coef_.shape == (n_stimuli, n_voxels)


@pytest.mark.parametrize('chunk_axis', [1, 2, None])
def test_chunk_axis_does_not_change_the_prediction(dataset, tmp_path,
                                                   chunk_axis):
    """``chunk_axis`` blocks the computation; it must not move the numbers."""
    layer, subject, roi = 'conv_like', 'sub-01', 'VC'

    reference_decoder = str(tmp_path / 'decoders_ref')
    reference_decoded = str(tmp_path / 'decoded_ref')
    pipeline.run_training(dataset, reference_decoder, alpha=ALPHA,
                          analysis_name='train_ref')
    pipeline.run_prediction(dataset, reference_decoder, reference_decoded,
                            chunk_axis=None, layers=[layer],
                            analysis_name='predict_ref')
    reference = pipeline.read_decoded_features(
        reference_decoded, layer, subject, roi, dataset.unique_test_labels)

    decoder_dir = str(tmp_path / ('decoders_%s' % chunk_axis))
    decoded_dir = str(tmp_path / ('decoded_%s' % chunk_axis))
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          analysis_name='train_%s' % chunk_axis)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=chunk_axis, layers=[layer],
                            analysis_name='predict_%s' % chunk_axis)
    actual = pipeline.read_decoded_features(
        decoded_dir, layer, subject, roi, dataset.unique_test_labels)

    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('chunk_axis', [1, 2, 3])
def test_combine_features_matches_the_unblocked_contraction(chunk_axis):
    """Blocking contracts over the same axis, to within rounding.

    The contraction runs over the stimulus axis only, so every output element is
    mathematically independent of the blocking.  It is not guaranteed to be
    bit-identical: BLAS may accumulate a width-1 slice in a different order than
    the full array, which shows up in float32 (see the float32 case below).
    """
    rng = np.random.RandomState(0)
    coefficients = rng.randn(5, 7)
    features = rng.randn(7, 3, 4, 2)

    whole = combine_features(coefficients, features, chunk_axis=None)
    blocked = combine_features(coefficients, features, chunk_axis=chunk_axis)

    assert blocked.shape == whole.shape
    np.testing.assert_allclose(blocked, whole, rtol=1e-12, atol=1e-12)


def test_combine_features_blocking_error_stays_at_rounding_level():
    """At realistic float32 sizes the two orders differ only by rounding."""
    rng = np.random.RandomState(0)
    coefficients = rng.randn(50, 600).astype(np.float32)
    features = rng.randn(600, 512, 14, 14).astype(np.float32)

    whole = combine_features(coefficients, features, chunk_axis=None)
    blocked = combine_features(coefficients, features, chunk_axis=1)

    np.testing.assert_allclose(blocked, whole, rtol=1e-5,
                               atol=1e-5 * np.max(np.abs(whole)))


def test_combine_features_ignores_chunk_axis_for_2d_features():
    """2-D features are below bdpy's chunking threshold, as before."""
    rng = np.random.RandomState(0)
    coefficients = rng.randn(4, 6)
    features = rng.randn(6, 9)

    np.testing.assert_array_equal(
        combine_features(coefficients, features, chunk_axis=1),
        combine_features(coefficients, features, chunk_axis=None))


def test_each_feature_channel_depends_only_on_its_own_features(dataset,
                                                               tmp_path):
    """A per-position shift of one channel reappears at exactly those positions.

    The perturbation has to differ *within* the channel. A uniform shift of the
    whole channel would still pass if a flatten/unflatten pair disagreed about C
    versus Fortran order, because a transposition of identical values is
    invisible; with distinct values per position, a mismatch would move the
    response to the wrong position inside the channel.

    A constant per-position shift passes through the decoder exactly: it moves
    ``y_mean`` by ``delta`` and leaves ``y_norm`` untouched, so the normalized
    target is unchanged and the un-normalization adds ``delta`` back. In the
    factorized decoder the same follows from the stimulus coefficients summing
    to one.
    """
    import shutil

    from bdpy.dataform import load_array, save_array

    layer, subject, roi = 'conv_like', 'sub-01', 'VC'

    baseline_decoder = str(tmp_path / 'decoders_base')
    baseline_decoded = str(tmp_path / 'decoded_base')
    pipeline.run_training(dataset, baseline_decoder, alpha=ALPHA, analysis_name='train_base')
    pipeline.run_prediction(dataset, baseline_decoder, baseline_decoded,
                            chunk_axis=1, layers=[layer],
                            analysis_name='predict_base')
    baseline = pipeline.read_decoded_features(
        baseline_decoded, layer, subject, roi, dataset.unique_test_labels)

    # Copy the feature tree and perturb channel 1 only, by a different amount at
    # each position within the channel.
    delta = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32)
    assert delta.shape == dataset.layer_shapes[layer][1:]
    assert not np.allclose(delta, delta.T), \
        'delta must be asymmetric, or an order mix-up would be invisible'

    perturbed_features = str(tmp_path / 'features_perturbed')
    shutil.copytree(dataset.train_features_dir, perturbed_features)
    for label in dataset.unique_train_labels:
        path = os.path.join(perturbed_features, layer, '%s.mat' % label)
        array = load_array(path, key='feat')
        array[:, 1] = array[:, 1] + delta
        save_array(path, array, key='feat', dtype=np.float32, sparse=False)

    dataset.train_features_dir = perturbed_features
    perturbed_decoder = str(tmp_path / 'decoders_perturbed')
    perturbed_decoded = str(tmp_path / 'decoded_perturbed')
    pipeline.run_training(dataset, perturbed_decoder, alpha=ALPHA,
                          analysis_name='train_perturbed')
    pipeline.run_prediction(dataset, perturbed_decoder, perturbed_decoded,
                            chunk_axis=1, layers=[layer],
                            analysis_name='predict_perturbed')
    perturbed = pipeline.read_decoded_features(
        perturbed_decoded, layer, subject, roi, dataset.unique_test_labels)

    shift = perturbed - baseline

    # Channel 1 moves by delta, position for position.
    np.testing.assert_allclose(
        shift[:, 1], np.broadcast_to(delta, shift[:, 1].shape),
        rtol=0, atol=1e-4)

    # Nothing else moves.
    other = np.delete(shift, 1, axis=1)
    np.testing.assert_allclose(other, np.zeros_like(other), rtol=0, atol=1e-4)
