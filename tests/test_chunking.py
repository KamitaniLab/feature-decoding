"""Chunking of the feature dimension.

bdpy chunks the target along ``chunk_axis`` when the feature array is at least
3-D (``Y.ndim >= chunk_ndim + 1``, ``chunk_ndim=2``), fitting one model per
index along that axis and flattening/unflattening each chunk with
``order='F'``.  The shipped config only enables ``fc6``/``fc7``/``fc8``, which
are 2-D and never reach this path, so it is pinned here explicitly.
"""

from __future__ import annotations

import os
import pickle

import numpy as np

from tests.helpers import legacy_ridge, pipeline

ALPHA = 100


def _model_files(decoder_dir, layer, subject='sub-01', roi='VC'):
    directory = pipeline.model_dir(decoder_dir, layer, subject, roi)
    return sorted(f for f in os.listdir(directory) if f.endswith('.pkl.gz'))


def test_multidimensional_features_are_chunked(dataset, decoder_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA, chunk_axis=1,
                          layers=['conv_like'])

    n_chunks = dataset.layer_shapes['conv_like'][0]
    assert _model_files(decoder_dir, 'conv_like') == [
        '%08d.pkl.gz' % i for i in range(n_chunks)]


def test_two_dimensional_features_are_not_chunked(dataset, decoder_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA, chunk_axis=1,
                          layers=['fc_like'])

    assert _model_files(decoder_dir, 'fc_like') == ['model.pkl.gz']


def test_chunk_axis_none_trains_a_single_model(dataset, decoder_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA, chunk_axis=None,
                          layers=['conv_like'])

    assert _model_files(decoder_dir, 'conv_like') == ['model.pkl.gz']


def test_saved_chunk_shape_and_fortran_order(dataset, decoder_dir):
    """Each chunk records ``y_shape`` with the chunk axis kept at size 1."""
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA, chunk_axis=1,
                          layers=['conv_like'])

    shape = dataset.layer_shapes['conv_like']
    directory = pipeline.model_dir(decoder_dir, 'conv_like', 'sub-01', 'VC')
    for name in _model_files(decoder_dir, 'conv_like'):
        with open(os.path.join(directory, name), 'rb') as f:
            payload = pickle.load(f)
        assert payload['y_shape'] == (1,) + shape[1:]
        # The fitted model is flat over the chunk's units.
        assert payload['model'].coef_.shape[0] == int(np.prod(shape[1:]))


def test_chunked_prediction_matches_unchunked(dataset, tmp_path):
    """``chunk_axis`` is a memory device: it must not change the result."""
    layer, subject, roi = 'conv_like', 'sub-01', 'VC'

    results = {}
    for tag, chunk_axis in (('chunked1', 1), ('chunked2', 2), ('whole', None)):
        decoder_dir = str(tmp_path / ('decoders_%s' % tag))
        decoded_dir = str(tmp_path / ('decoded_%s' % tag))
        pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                              chunk_axis=chunk_axis, layers=[layer],
                              analysis_name='train_%s' % tag)
        pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                                chunk_axis=chunk_axis, layers=[layer],
                                analysis_name='predict_%s' % tag)
        results[tag] = pipeline.read_decoded_features(
            decoded_dir, layer, subject, roi, dataset.unique_test_labels)

    np.testing.assert_allclose(results['chunked1'], results['whole'],
                               rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(results['chunked2'], results['whole'],
                               rtol=1e-4, atol=1e-5)


def test_each_chunk_depends_only_on_its_own_features(dataset, tmp_path):
    """A per-position shift of one channel reappears at exactly those positions.

    This is what makes the ``order='F'`` reshape round trip observable. The
    perturbation has to differ *within* the chunk: a uniform shift of the whole
    channel would still pass if the flatten and unflatten disagreed about C
    versus Fortran order, because a transposition of identical values is
    invisible. With distinct values per position, a mismatched pair would move
    the response to the wrong position inside the channel.

    A constant per-position shift passes through the decoder exactly: it moves
    ``y_mean`` by ``delta`` and leaves ``y_norm`` untouched, so the normalized
    target is unchanged and the un-normalization adds ``delta`` back.
    """
    import shutil

    from bdpy.dataform import load_array, save_array

    layer, subject, roi = 'conv_like', 'sub-01', 'VC'

    baseline_decoder = str(tmp_path / 'decoders_base')
    baseline_decoded = str(tmp_path / 'decoded_base')
    pipeline.run_training(dataset, baseline_decoder, alpha=ALPHA, chunk_axis=1,
                          layers=[layer], analysis_name='train_base')
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
                          chunk_axis=1, layers=[layer],
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
    np.testing.assert_allclose(other, np.zeros_like(other),
                               rtol=0, atol=1e-4)
