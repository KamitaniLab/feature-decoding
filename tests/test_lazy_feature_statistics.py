"""The training-feature statistics: compatibility sidecars, written lazily.

``evaluation.py`` normalizes the pattern correlation with the per-unit mean and
SD of the training features, read from
``<decoder>/<layer>/<subject>/<roi>/model/``.  The factorized decoder cannot
produce them: training never opens a feature file, and prediction does not need
them either (the feature normalization cancels).  So ``predict_feature.py`` --
the step that has the features -- writes them there, purely for the unchanged
evaluation pipeline.

Three things make that work: the write happens *before* the "results directory
already exists" skip, so a run whose features were already decoded still gets
them; it is atomic, because parallel prediction workers are a supported mode and
two runs with different analysis names are not excluded by the DistComp lock;
and the statistics are computed once per ``(layer, ordered training labels)``
and then written into every decoder directory that needs them.  Existing files
are never overwritten: one decoder directory belongs to one training feature
configuration.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest
from bdpy.dataform import load_array, save_array

import predict_feature
import ridge_factorization
from ridge_factorization import save_array_atomic
from tests.helpers import pipeline

ALPHA = 100
CHUNK_AXIS = 1
LAYER = 'fc_like'
ONE_SUBJECT = ['sub-01']
ONE_ROI = {'VC': 'ROI_VC = 1'}
KEYS = ('y_mean', 'y_norm')


def _train(dataset, decoder_dir, **kwargs):
    return pipeline.run_training(
        dataset, decoder_dir, alpha=ALPHA, subjects=ONE_SUBJECT, rois=ONE_ROI, **kwargs)


def _predict(dataset, decoder_dir, decoded_dir, **kwargs):
    return pipeline.run_prediction(
        dataset, decoder_dir, decoded_dir, chunk_axis=CHUNK_AXIS,
        layers=[LAYER], subjects=ONE_SUBJECT, rois=ONE_ROI, **kwargs)


def _statistics_paths(decoder_dir, roi='VC'):
    directory = pipeline.layer_model_dir(decoder_dir, LAYER, 'sub-01', roi)
    return {key: os.path.join(directory, '%s.mat' % key) for key in KEYS}


# The materialization ##########################################################

def test_a_second_prediction_run_does_not_rewrite_the_statistics(dataset,
                                                                 tmp_path):
    """The cheap existence check comes first, so nothing is recomputed."""
    decoder_dir = str(tmp_path / 'decoders')
    decoded_dir = str(tmp_path / 'decoded')
    _train(dataset, decoder_dir)
    _predict(dataset, decoder_dir, decoded_dir)

    paths = _statistics_paths(decoder_dir)
    before = {key: os.stat(path).st_mtime_ns for key, path in paths.items()}

    _predict(dataset, decoder_dir, decoded_dir, analysis_name='again')

    after = {key: os.stat(path).st_mtime_ns for key, path in paths.items()}
    assert after == before


def test_the_statistics_are_materialized_past_the_already_done_skip(
        dataset, tmp_path, capsys):
    """The reason the write runs before the skip.

    A prediction whose features were already decoded would otherwise skip
    forever, leaving ``evaluation.py`` permanently unable to run against it.
    """
    decoder_dir = str(tmp_path / 'decoders')
    decoded_dir = str(tmp_path / 'decoded')
    _train(dataset, decoder_dir)
    _predict(dataset, decoder_dir, decoded_dir)

    paths = _statistics_paths(decoder_dir)
    for path in paths.values():
        os.unlink(path)

    decoded_before = sorted(
        (name, os.stat(os.path.join(decoded_dir, LAYER, 'sub-01', 'VC',
                                    name)).st_mtime_ns)
        for name in os.listdir(os.path.join(decoded_dir, LAYER, 'sub-01',
                                            'VC')))

    capsys.readouterr()
    _predict(dataset, decoder_dir, decoded_dir, analysis_name='rerun')

    # Every (layer, subject, ROI) hit the skip ...
    assert 'is already done. Skipped.' in capsys.readouterr().out
    decoded_after = sorted(
        (name, os.stat(os.path.join(decoded_dir, LAYER, 'sub-01', 'VC',
                                    name)).st_mtime_ns)
        for name in os.listdir(os.path.join(decoded_dir, LAYER, 'sub-01',
                                            'VC')))
    assert decoded_after == decoded_before

    # ... and the statistics came back anyway, with the right values.
    feat = dataset.feature_matrix(LAYER, dataset.unique_train_labels)
    np.testing.assert_allclose(
        load_array(paths['y_mean'], key='y_mean'),
        np.mean(feat, axis=0)[np.newaxis].astype(np.float32),
        rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        load_array(paths['y_norm'], key='y_norm'),
        np.std(feat, axis=0, ddof=1)[np.newaxis].astype(np.float32),
        rtol=1e-5, atol=1e-6)


def test_a_finished_prediction_reruns_without_the_training_features(dataset,
                                                                    tmp_path):
    """Nothing left to decode, so nothing to verify or materialize either."""
    decoder_dir = str(tmp_path / 'decoders')
    decoded_dir = str(tmp_path / 'decoded')
    _train(dataset, decoder_dir)
    _predict(dataset, decoder_dir, decoded_dir)

    _predict(dataset, decoder_dir, decoded_dir, features_paths=[],
             analysis_name='no_features')


def test_an_unwritable_output_directory_warns_instead_of_failing(dataset,
                                                                 tmp_path,
                                                                 monkeypatch):
    """Decoding must still work when the statistics cannot be written.

    The write is made to fail here rather than by permission bits, since the
    test suite may run as a user for whom no permission bit denies anything.
    """
    decoder_dir = str(tmp_path / 'decoders')
    decoded_dir = str(tmp_path / 'decoded')
    _train(dataset, decoder_dir)

    def read_only(*args, **kwargs):
        raise IOError('Read-only file system')

    monkeypatch.setattr(predict_feature, 'save_array_atomic', read_only)
    with pytest.warns(UserWarning, match='read-only'):
        _predict(dataset, decoder_dir, decoded_dir)

    assert not os.path.exists(_statistics_paths(decoder_dir)['y_mean'])
    assert pipeline.read_decoded_features(
        decoded_dir, LAYER, 'sub-01', 'VC',
        dataset.unique_test_labels).shape[0] == len(dataset.unique_test_labels)


def test_the_statistics_are_computed_once_per_layer_and_label_set(
        dataset, decoder_dir, decoded_dir, monkeypatch):
    """Not once per decoder: they do not depend on the ROI.

    At DeepRecon scale each computation is two passes over a 15 GB tensor, and
    there are 9 ROIs per subject.
    """
    computed = []
    original = ridge_factorization.feature_statistics

    def spy(features):
        computed.append(features.shape)
        return original(features)

    monkeypatch.setattr(predict_feature, 'feature_statistics', spy)

    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS)

    # Both synthetic subjects see the same stimuli, so one computation per
    # layer covers every (subject, ROI) of the run ...
    assert len(dataset.rois) > 1 and len(dataset.test_fmri) > 1
    assert len(computed) == len(dataset.layers)

    # ... while every decoder directory still gets its own copy, because that
    # is where evaluation.py looks.
    for layer in dataset.layers:
        for subject in dataset.test_fmri:
            for roi in dataset.rois:
                assert pipeline.read_feature_statistics(decoder_dir, layer,
                                                        subject, roi)


def test_existing_statistics_are_never_overwritten(dataset, tmp_path):
    """One decoder directory belongs to one training feature configuration.

    Re-pointing it at another ``Y_tr`` or feature index is not a supported
    workflow, so prediction fills the sidecars in when they are missing and
    leaves them alone otherwise -- it never decides that what is there is
    wrong.
    """
    decoder_dir = str(tmp_path / 'decoders')
    _train(dataset, decoder_dir)
    _predict(dataset, decoder_dir, str(tmp_path / 'decoded'))

    paths = _statistics_paths(decoder_dir)
    planted = np.full((1, 3), 7.0, dtype=np.float32)
    for path, key in ((paths[key], key) for key in KEYS):
        save_array(path, planted, key=key, dtype=np.float32, sparse=False)

    _predict(dataset, decoder_dir, str(tmp_path / 'decoded_again'),
             analysis_name='again')

    for key in KEYS:
        np.testing.assert_array_equal(load_array(paths[key], key=key), planted)


# save_array_atomic ############################################################

def _temp_files(directory):
    return [name for name in os.listdir(directory) if name.startswith('.tmp')]


def test_atomic_save_writes_a_loadable_file(tmp_path):
    """The temp name has to keep the '.mat' suffix that hdf5storage appends."""
    path = str(tmp_path / 'y_mean.mat')
    array = np.arange(6, dtype=np.float32)[np.newaxis]

    save_array_atomic(path, array, key='y_mean', dtype=np.float32)

    np.testing.assert_array_equal(load_array(path, key='y_mean'), array)
    assert _temp_files(str(tmp_path)) == []


def test_atomic_save_replaces_an_existing_file(tmp_path):
    path = str(tmp_path / 'y_mean.mat')
    save_array_atomic(path, np.zeros((1, 6), dtype=np.float32), key='y_mean')

    replacement = np.arange(6, dtype=np.float32)[np.newaxis]
    save_array_atomic(path, replacement, key='y_mean')

    np.testing.assert_array_equal(load_array(path, key='y_mean'), replacement)
    assert _temp_files(str(tmp_path)) == []


def test_a_failed_atomic_save_leaves_the_original_intact(tmp_path,
                                                         monkeypatch):
    path = str(tmp_path / 'y_mean.mat')
    original = np.arange(6, dtype=np.float32)[np.newaxis]
    save_array_atomic(path, original, key='y_mean')

    def broken(*args, **kwargs):
        raise IOError('disk full')

    monkeypatch.setattr(ridge_factorization, 'save_array', broken)
    with pytest.raises(IOError):
        save_array_atomic(path, np.zeros((1, 6), dtype=np.float32),
                          key='y_mean')

    np.testing.assert_array_equal(load_array(path, key='y_mean'), original)
    assert _temp_files(str(tmp_path)) == []


def test_a_partially_written_temp_file_is_cleaned_up(tmp_path, monkeypatch):
    """The failure that would otherwise leave rubbish next to the target."""
    path = str(tmp_path / 'y_mean.mat')
    real_save = ridge_factorization.save_array

    def write_then_fail(save_file, array, **kwargs):
        real_save(save_file, array, **kwargs)
        raise IOError('interrupted after the write')

    monkeypatch.setattr(ridge_factorization, 'save_array', write_then_fail)
    with pytest.raises(IOError):
        save_array_atomic(path, np.zeros((1, 6), dtype=np.float32),
                          key='y_mean')

    assert not os.path.exists(path)
    assert _temp_files(str(tmp_path)) == []


def test_concurrent_atomic_saves_leave_one_valid_file(tmp_path):
    """Two writers past the existence check, as parallel workers can be.

    Each writer builds its own temp name, so the loser's rename is the only
    thing that races, and a reader sees one complete file either way.
    """
    path = str(tmp_path / 'y_norm.mat')
    candidates = [np.full((1, 6), value, dtype=np.float32)
                  for value in (1.0, 2.0)]
    start = threading.Barrier(len(candidates))
    errors = []

    def write(array):
        try:
            start.wait()
            save_array_atomic(path, array, key='y_norm')
        except BaseException as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=write, args=(array,))
               for array in candidates]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    loaded = load_array(path, key='y_norm')
    assert any(np.array_equal(loaded, array) for array in candidates)
    assert _temp_files(str(tmp_path)) == []
