"""The factorized decoder artifact: contents, layer independence, robustness."""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

import ridge_factorization
from tests.helpers import pipeline, synthetic

ALPHA = 100
ONE_SUBJECT = ['sub-01']
ONE_ROI = {'VC': 'ROI_VC = 1'}


def test_the_artifact_carries_the_model_and_its_stimulus_order(dataset,
                                                               decoder_dir):
    """One file holds everything the decoder is.

    The coefficients are indexed by training stimulus, so the ordered labels
    live in the same pickle: they cannot be reordered, renamed or replaced
    independently of the model they index.  Nothing about the features is
    recorded -- training never reads one -- and not even a layer, since the
    model is shared by all of them.
    """
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    directory = pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC')
    artifact = ridge_factorization.load_factorized_model(directory)
    expected = [str(lb) for lb in np.unique(dataset.labels('sub-01', 'train'))]

    assert artifact['format'] == ridge_factorization.FORMAT_NAME
    assert artifact['version'] == ridge_factorization.FORMAT_VERSION
    assert artifact['train_labels'] == expected
    assert artifact['y_shape'] == (len(expected),)
    assert artifact['model'].coef_.shape[0] == len(expected)

    assert artifact['alpha'] == float(ALPHA)
    assert artifact['subject'] == 'sub-01'
    assert artifact['roi'] == 'VC'
    assert artifact['dtype'] == 'float32'
    assert artifact['n_trials'] == len(dataset.labels('sub-01', 'train'))
    assert artifact['n_voxels'] == dataset.brain('sub-01', 'VC').shape[1]

    for absent in ('layer', 'chunk_axis', 'feature_source', 'feature_shape',
                   'feature_digest', 'feature_index_file'):
        assert absent not in artifact, absent


def test_the_brain_side_model_is_stored_once(dataset, decoder_dir,
                                             decoded_dir):
    """The stimulus-basis Ridge does not depend on the layer, so it is not
    copied under every layer -- the per-layer directories that prediction
    creates hold only the statistics ``evaluation.py`` reads.
    """
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir)

    for subject in dataset.train_fmri:
        for roi in dataset.rois:
            assert sorted(os.listdir(
                pipeline.brain_model_dir(decoder_dir, subject, roi))) == [
                    'info.yaml', 'model.pkl.gz', 'x_mean.mat', 'x_norm.mat']
            for layer in dataset.layers:
                sidecars = pipeline.layer_model_dir(decoder_dir, layer,
                                                    subject, roi)
                assert sorted(os.listdir(sidecars)) == ['y_mean.mat',
                                                        'y_norm.mat']


def test_model_size_does_not_grow_with_the_feature_dimension(dataset,
                                                             decoder_dir,
                                                             decoded_dir,
                                                             tmp_path):
    """The stored model scales with n_train_stimuli x n_voxels only.

    This is the whole point of the factorization: the direct implementation
    stored n_features x n_voxels coefficients, so a 50x larger layer meant a
    50x larger model file. Here the layer does not enter the model at all --
    adding one changes the sidecars and nothing else.
    """
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)
    before = os.path.getsize(os.path.join(
        pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC'), 'model.pkl.gz'))

    small_units = int(np.prod(dataset.layer_shapes['fc_like']))
    synthetic.add_layer(dataset, 'big_like', (small_units * 50,))

    wider_dir = str(tmp_path / 'decoders_with_a_big_layer')
    pipeline.run_training(dataset, wider_dir, alpha=ALPHA,
                          analysis_name='training_wide')
    after = os.path.getsize(os.path.join(
        pipeline.brain_model_dir(wider_dir, 'sub-01', 'VC'), 'model.pkl.gz'))
    assert after == before

    pipeline.run_prediction(dataset, wider_dir, decoded_dir,
                            layers=['fc_like', 'big_like'])
    statistics = {
        layer: pipeline.read_feature_statistics(wider_dir, layer, 'sub-01',
                                                'VC')['y_mean'].size
        for layer in ('fc_like', 'big_like')}
    assert statistics['big_like'] == 50 * statistics['fc_like']


def test_the_format_version_is_checked(dataset, decoder_dir):
    """An artifact from a future version is refused, not misread."""
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          subjects=ONE_SUBJECT, rois=ONE_ROI)

    directory = pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC')
    path = ridge_factorization.model_file_path(directory)
    with open(path, 'rb') as f:
        artifact = pickle.load(f)
    artifact['version'] = ridge_factorization.FORMAT_VERSION + 1
    with open(path, 'wb') as f:
        pickle.dump(artifact, f)

    with pytest.raises(ValueError, match='Unsupported'):
        ridge_factorization.load_factorized_model(directory)
