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


def test_creates_one_model_dir_per_subject_and_roi(dataset, decoder_dir):
    """One brain-side model per (subject, ROI), and no per-layer copy of it.

    The fit is on the training stimulus basis, which does not depend on the
    layer, so there is nothing layer-specific to store.
    """
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    for subject in dataset.train_fmri:
        for roi in dataset.rois:
            directory = pipeline.brain_model_dir(decoder_dir, subject, roi)
            assert os.path.isdir(directory), directory
            for key in pipeline.BRAIN_NORM_KEYS:
                assert os.path.isfile(os.path.join(directory, key + '.mat'))
            for key in pipeline.FEATURE_NORM_KEYS:
                assert not os.path.exists(os.path.join(directory,
                                                       key + '.mat'))
            assert os.path.isfile(os.path.join(directory, 'info.yaml'))

    # Training does not create the per-layer directories at all: they hold the
    # statistics sidecars, which prediction writes.
    assert sorted(os.listdir(decoder_dir)) == sorted(dataset.train_fmri)


def test_normalization_parameter_shapes_and_dtype(dataset, decoder_dir):
    """Only the brain half: the decoder stores no feature statistics."""
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    for subject in dataset.train_fmri:
        for roi in dataset.rois:
            params = pipeline.read_norm_params(decoder_dir, subject, roi)
            n_voxels = dataset.brain(subject, roi).shape[1]
            assert params['x_mean'].shape == (1, n_voxels)
            assert params['x_norm'].shape == (1, n_voxels)
            for key in pipeline.BRAIN_NORM_KEYS:
                assert params[key].dtype == np.float32


def test_brain_normalization_uses_ddof_one_over_all_trials(dataset,
                                                           decoder_dir):
    """Brain mean/SD are per-voxel over *every* trial, with ``ddof=1``."""
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    subject, roi = 'sub-01', 'VC'
    brain = dataset.brain(subject, roi, 'train')
    params = pipeline.read_norm_params(decoder_dir, subject, roi,
                                       keys=pipeline.BRAIN_NORM_KEYS)

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


def test_feature_statistics_are_over_unique_stimuli(dataset, decoder_dir,
                                                    decoded_dir):
    """``y_mean``/``y_norm`` weight every stimulus once, not once per trial.

    The repetitions in this dataset are unbalanced, so the two definitions
    genuinely differ and the distinction is observable.  Prediction writes
    them, into the per-layer directory ``evaluation.py`` reads, because
    training never opens a feature file.
    """
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir)

    subject = 'sub-01'
    brain_labels = dataset.labels(subject, 'train')
    unique_labels = list(np.unique(brain_labels))

    for layer in dataset.layers:
        params = pipeline.read_feature_statistics(decoder_dir, layer, subject,
                                                  'VC')
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
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    roi = 'VC'
    params = pipeline.read_norm_params(decoder_dir, 'sub-01', roi,
                                       keys=pipeline.BRAIN_NORM_KEYS)
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
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    for subject in dataset.train_fmri:
        for roi in dataset.rois:
            info_file = os.path.join(
                pipeline.brain_model_dir(decoder_dir, subject, roi),
                'info.yaml')
            with open(info_file) as f:
                info = yaml.safe_load(f)
            assert info['_status']['computation_status'] == 'done'


def test_completed_analyses_are_skipped_on_rerun(dataset, decoder_dir):
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    watched = []
    for subject in dataset.train_fmri:
        for roi in dataset.rois:
            directory = pipeline.brain_model_dir(decoder_dir, subject, roi)
            for name in sorted(os.listdir(directory)):
                path = os.path.join(directory, name)
                watched.append((path, os.stat(path).st_mtime_ns))

    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          analysis_name='test_training_rerun')

    for path, mtime in watched:
        assert os.stat(path).st_mtime_ns == mtime, \
            '%s was rewritten although the analysis was already done' % path


def test_model_artifacts_are_readable(dataset, decoder_dir):
    """The brain-side ``model/`` holds a loadable model file."""
    import pickle

    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    directory = pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC')
    model_files = sorted(f for f in os.listdir(directory)
                         if f.endswith('.pkl.gz'))
    assert model_files
    for name in model_files:
        with open(os.path.join(directory, name), 'rb') as f:
            payload = pickle.load(f)
        assert 'model' in payload and 'y_shape' in payload
