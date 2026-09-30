"""Behavior of ``train_decoder_sklearn_ridge.featdec_sklearn_ridge_train``.

These tests pin the decoder *artifact contract* -- directory layout, file names,
array keys, dtypes and the exact definition of the saved normalization
parameters -- independently of how the regression itself is computed.
``evaluation.py`` reads ``y_mean.mat``/``y_norm.mat`` straight out of
``<decoder>/<layer>/<subject>/<roi>/model/``, so this layout is a public
interface, not an implementation detail.
"""

from __future__ import annotations

import os

import numpy as np
import yaml

from tests.helpers import pipeline

ALPHA = 100
CHUNK_AXIS = 1


def test_creates_model_dir_per_layer_subject_roi(dataset, decoder_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS)

    for layer in dataset.layers:
        for subject in dataset.train_fmri:
            for roi in dataset.rois:
                directory = pipeline.model_dir(decoder_dir, layer, subject,
                                               roi)
                assert os.path.isdir(directory), directory
                for key in pipeline.NORM_KEYS:
                    assert os.path.isfile(os.path.join(directory,
                                                       key + '.mat'))
                assert os.path.isfile(os.path.join(directory, 'info.yaml'))


def test_normalization_parameter_shapes_and_dtype(dataset, decoder_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS)

    for layer in dataset.layers:
        feature_shape = dataset.layer_shapes[layer]
        for subject in dataset.train_fmri:
            for roi in dataset.rois:
                params = pipeline.read_norm_params(decoder_dir, layer, subject,
                                                   roi)
                n_voxels = dataset.brain(subject, roi).shape[1]
                assert params['x_mean'].shape == (1, n_voxels)
                assert params['x_norm'].shape == (1, n_voxels)
                assert params['y_mean'].shape == (1,) + feature_shape
                assert params['y_norm'].shape == (1,) + feature_shape
                for key in pipeline.NORM_KEYS:
                    assert params[key].dtype == np.float32


def test_brain_normalization_uses_ddof_one_over_all_trials(dataset,
                                                           decoder_dir):
    """Brain mean/SD are per-voxel over *every* trial, with ``ddof=1``."""
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS)

    subject, roi, layer = 'sub-01', 'VC', dataset.layers[0]
    brain = dataset.brain(subject, roi, 'train')
    params = pipeline.read_norm_params(decoder_dir, layer, subject, roi)

    np.testing.assert_allclose(
        params['x_mean'], np.mean(brain, axis=0)[np.newaxis, :].astype(
            np.float32), rtol=1e-6)
    np.testing.assert_allclose(
        params['x_norm'],
        np.std(brain, axis=0, ddof=1)[np.newaxis, :].astype(np.float32),
        rtol=1e-6)
    # ddof=0 would be a different number for this data.
    assert not np.allclose(params['x_norm'],
                           np.std(brain, axis=0, ddof=0)[np.newaxis, :],
                           rtol=1e-4)


def test_feature_statistics_are_over_unique_stimuli(dataset, decoder_dir):
    """``y_mean``/``y_norm`` weight every stimulus once, not once per trial.

    The repetitions in this dataset are unbalanced, so the two definitions
    genuinely differ and the distinction is observable.
    """
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS)

    subject, roi = 'sub-01', 'VC'
    brain_labels = dataset.labels(subject, 'train')
    unique_labels = list(np.unique(brain_labels))

    for layer in dataset.layers:
        params = pipeline.read_norm_params(decoder_dir, layer, subject, roi)
        unique_feat = dataset.feature_matrix(layer, unique_labels)
        trial_feat = dataset.feature_matrix(layer, brain_labels)

        np.testing.assert_allclose(
            params['y_mean'],
            np.mean(unique_feat, axis=0)[np.newaxis].astype(np.float32),
            rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            params['y_norm'],
            np.std(unique_feat, axis=0, ddof=1)[np.newaxis].astype(np.float32),
            rtol=1e-5, atol=1e-6)
        assert not np.allclose(
            params['y_mean'], np.mean(trial_feat, axis=0)[np.newaxis],
            rtol=1e-4)


def test_multi_file_subject_uses_every_file(dataset, decoder_dir):
    """sub-01's trials are split over two ``.h5`` files; both must be used."""
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS)

    layer, roi = dataset.layers[0], 'VC'
    params = pipeline.read_norm_params(decoder_dir, layer, 'sub-01', roi)
    all_trials = dataset.brain('sub-01', roi, 'train')

    np.testing.assert_allclose(
        params['x_mean'],
        np.mean(all_trials, axis=0)[np.newaxis, :].astype(np.float32),
        rtol=1e-6)
    # Only the first file would give a different mean.
    assert not np.allclose(params['x_mean'],
                           np.mean(all_trials[:7], axis=0)[np.newaxis, :],
                           rtol=1e-4)


def test_info_yaml_marks_completion(dataset, decoder_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS)

    for layer in dataset.layers:
        for subject in dataset.train_fmri:
            for roi in dataset.rois:
                info_file = os.path.join(
                    pipeline.model_dir(decoder_dir, layer, subject, roi),
                    'info.yaml')
                with open(info_file) as f:
                    info = yaml.safe_load(f)
                assert info['_status']['computation_status'] == 'done'


def test_completed_analyses_are_skipped_on_rerun(dataset, decoder_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS)

    watched = []
    for layer in dataset.layers:
        for subject in dataset.train_fmri:
            for roi in dataset.rois:
                directory = pipeline.model_dir(decoder_dir, layer, subject,
                                               roi)
                for name in sorted(os.listdir(directory)):
                    path = os.path.join(directory, name)
                    watched.append((path, os.stat(path).st_mtime_ns))

    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS,
                          analysis_name='test_training_rerun')

    for path, mtime in watched:
        assert os.stat(path).st_mtime_ns == mtime, \
            '%s was rewritten although the analysis was already done' % path


def test_model_artifacts_are_readable(dataset, decoder_dir):
    """Every ``model/`` directory holds at least one loadable model file."""
    import pickle

    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          chunk_axis=CHUNK_AXIS)

    for layer in dataset.layers:
        directory = pipeline.model_dir(decoder_dir, layer, 'sub-01', 'VC')
        model_files = sorted(f for f in os.listdir(directory)
                             if f.endswith('.pkl.gz'))
        assert model_files
        for name in model_files:
            with open(os.path.join(directory, name), 'rb') as f:
                payload = pickle.load(f)
            assert 'model' in payload and 'y_shape' in payload
