"""The factorized decoder artifact: metadata, layer independence, robustness."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import yaml

import ridge_factorization
from tests.helpers import pipeline, synthetic

ALPHA = 100
ONE_SUBJECT = ['sub-01']
ONE_ROI = {'VC': 'ROI_VC = 1'}


def test_metadata_describes_the_format(dataset, decoder_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    directory = pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC')
    with open(os.path.join(directory, 'factorized.yaml')) as f:
        meta = yaml.safe_load(f)

    assert meta['format'] == ridge_factorization.FORMAT_NAME
    assert meta['version'] == ridge_factorization.FORMAT_VERSION
    assert meta['alpha'] == float(ALPHA)
    assert meta['subject'] == 'sub-01'
    assert meta['roi'] == 'VC'
    assert meta['dtype'] == 'float32'
    assert meta['n_trials'] == len(dataset.labels('sub-01', 'train'))
    assert meta['n_voxels'] == dataset.brain('sub-01', 'VC').shape[1]
    assert meta['n_train_stimuli'] == len(
        np.unique(dataset.labels('sub-01', 'train')))

    # The decoder holds coefficients over the training stimuli, so the artifact
    # pins their identity and order -- and nothing else about the features,
    # which it never reads. Not even a layer: the model is shared by all of
    # them.
    labels = [str(lb) for lb in np.unique(dataset.labels('sub-01', 'train'))]
    assert meta['labels_digest_spec'] == \
        ridge_factorization.LABELS_DIGEST_SPEC
    assert meta['labels_digest'] == ridge_factorization.labels_digest(labels)

    for absent in ('layer', 'chunk_axis', 'feature_source', 'feature_shape',
                   'feature_digest', 'feature_digest_spec',
                   'feature_index_file'):
        assert absent not in meta, absent


def test_train_labels_record_the_feature_row_order(dataset, decoder_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    expected = [str(lb) for lb in np.unique(dataset.labels('sub-01', 'train'))]
    directory = pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC')
    with open(os.path.join(directory, 'train_labels.json')) as f:
        labels = json.load(f)
    assert labels == expected
    assert ridge_factorization.load_train_labels(directory) == expected


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
            assert os.path.isfile(os.path.join(
                pipeline.brain_model_dir(decoder_dir, subject, roi),
                'model.pkl.gz'))
            for layer in dataset.layers:
                sidecars = pipeline.layer_model_dir(decoder_dir, layer,
                                                    subject, roi)
                assert sorted(os.listdir(sidecars)) == ['y_mean.mat',
                                                        'y_norm.mat']


def test_model_size_does_not_grow_with_the_feature_dimension(dataset,
                                                             decoder_dir,
                                                             decoded_dir):
    """The stored model scales with n_train_stimuli x n_voxels only.

    This is the whole point of the factorization: the direct implementation
    stored n_features x n_voxels coefficients, so a 50x larger layer meant a
    50x larger model file. Here the layer does not enter the model at all --
    only the sidecars grow with it.
    """
    small_units = int(np.prod(dataset.layer_shapes['fc_like']))
    synthetic.add_layer(dataset, 'big_like', (small_units * 50,))

    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            layers=['fc_like', 'big_like'])

    directory = pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC')
    assert sorted(os.listdir(directory)) == [
        'factorized.yaml', 'info.yaml', 'model.pkl.gz', 'train_labels.json',
        'x_mean.mat', 'x_norm.mat',
    ]

    statistics = {
        layer: pipeline.read_feature_statistics(decoder_dir, layer, 'sub-01',
                                                'VC')['y_mean'].size
        for layer in ('fc_like', 'big_like')}
    assert statistics['big_like'] == 50 * statistics['fc_like']


def test_metadata_version_is_checked(dataset, decoder_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          subjects=ONE_SUBJECT, rois=ONE_ROI)

    directory = pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC')
    meta_path = os.path.join(directory, 'factorized.yaml')
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    meta['version'] = ridge_factorization.FORMAT_VERSION + 1
    with open(meta_path, 'w') as f:
        f.write(yaml.dump(meta))

    with pytest.raises(ValueError, match='Unsupported'):
        ridge_factorization.load_factorized_metadata(directory)

