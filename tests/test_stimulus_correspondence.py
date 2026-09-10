"""The decoder's only requirement about the features it is combined with.

The factorized fit is on the *stimulus basis*: it maps brain activity to one
coefficient per training stimulus and never sees a feature value.  What the
stored coefficients therefore require of the features they are combined with is
exactly one thing:

    coefficient column i is multiplied by the feature row of training stimulus i

These tests cover the three places that can break it -- the stored label order,
the row assembly, and the width of the model -- and that the decoder's own
training features go through them untouched.

(The algebra also holds for any other feature matrix with the same row order;
that is a property of `C @ F`, tested in ``tests/test_ridge_factorization.py``,
not a supported way to use a decoder *directory*, which carries one set of
feature statistics for ``evaluation.py``.)
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pytest
import yaml

import ridge_factorization
from ridge_factorization import (
    TrainingFeatureLoader,
    labels_digest,
    resolve_feature_index_path,
)
from tests.helpers import legacy_ridge, pipeline, synthetic

ALPHA = 100
CHUNK_AXIS = 1
ONE_SUBJECT = ['sub-01']
ONE_ROI = {'VC': 'ROI_VC = 1'}
LAYER = 'fc_like'


def _train(dataset, decoder_dir, **kwargs):
    return pipeline.run_training(
        dataset, decoder_dir, alpha=ALPHA, subjects=ONE_SUBJECT, rois=ONE_ROI, **kwargs)


def _predict(dataset, decoder_dir, decoded_dir, **kwargs):
    return pipeline.run_prediction(
        dataset, decoder_dir, decoded_dir, chunk_axis=CHUNK_AXIS,
        layers=[LAYER], subjects=ONE_SUBJECT, rois=ONE_ROI, **kwargs)


def _stimulus_coefficients(dataset, decoder_dir):
    """Run the decoder's brain-side model by hand, on averaged test trials."""
    directory = pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC')
    with open(ridge_factorization.model_file_path(directory), 'rb') as f:
        model = pickle.load(f)['model']
    params = pipeline.read_norm_params(decoder_dir, 'sub-01', 'VC')
    test_brain, _ = legacy_ridge.average_test_brain(
        dataset.brain('sub-01', 'VC', 'test'), dataset.labels('sub-01', 'test'))
    normalized = (test_brain - params['x_mean']) / params['x_norm']
    return model.predict(normalized.astype(np.float32))


# The invariant, positively #####################################################

def test_matching_features_are_accepted(dataset, decoder_dir, decoded_dir):
    """The checks must not reject the decoder's own training features."""
    _train(dataset, decoder_dir)
    _predict(dataset, decoder_dir, decoded_dir)

    brain_labels = dataset.labels('sub-01', 'train')
    feat_labels = list(np.unique(brain_labels))
    trained = legacy_ridge.legacy_train(
        dataset.brain('sub-01', 'VC', 'train'), brain_labels,
        dataset.feature_matrix(LAYER, feat_labels), feat_labels,
        alpha=ALPHA, chunk_axis=CHUNK_AXIS)
    test_brain, test_labels = legacy_ridge.average_test_brain(
        dataset.brain('sub-01', 'VC', 'test'), dataset.labels('sub-01', 'test'))

    np.testing.assert_allclose(
        pipeline.read_decoded_features(decoded_dir, LAYER, 'sub-01', 'VC',
                                       test_labels),
        legacy_ridge.legacy_predict(trained, test_brain).astype(np.float32),
        rtol=1e-5, atol=1e-6)


# The stored label order ########################################################

def test_permuted_train_labels_are_rejected(dataset, tmp_path):
    """The case that rules out per-column statistics as the guard.

    Reversing ``train_labels.json`` leaves every feature mean and SD untouched
    while pairing each coefficient column with the wrong stimulus, so only a
    digest of the *ordered* labels can see it -- and it needs no feature read.
    """
    decoder_dir = str(tmp_path / 'decoders')
    _train(dataset, decoder_dir)
    model_dir = pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC')
    labels_path = os.path.join(model_dir, 'train_labels.json')
    with open(labels_path) as f:
        labels = json.load(f)
    with open(labels_path, 'w') as f:
        json.dump(list(reversed(labels)), f)

    ordered = dataset.feature_matrix(LAYER, labels)
    permuted = dataset.feature_matrix(LAYER, list(reversed(labels)))
    # (Only the float32 summation order differs, hence a tolerance rather than
    # equality.)
    np.testing.assert_allclose(np.mean(permuted, axis=0),
                               np.mean(ordered, axis=0), rtol=1e-6)
    np.testing.assert_allclose(np.std(permuted, axis=0, ddof=1),
                               np.std(ordered, axis=0, ddof=1), rtol=1e-6)

    with pytest.raises(ValueError, match='do not match the order'):
        _predict(dataset, decoder_dir, str(tmp_path / 'decoded'),
                 analysis_name='permuted')


def test_a_renamed_train_label_is_rejected(dataset, tmp_path):
    """Editing one label keeps the count but changes which stimulus is meant."""
    decoder_dir = str(tmp_path / 'decoders')
    _train(dataset, decoder_dir)
    labels_path = os.path.join(
        pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC'),
        'train_labels.json')
    with open(labels_path) as f:
        labels = json.load(f)
    labels[0] = labels[-1]
    with open(labels_path, 'w') as f:
        json.dump(labels, f)

    with pytest.raises(ValueError, match='do not match the order'):
        _predict(dataset, decoder_dir, str(tmp_path / 'decoded'),
                 analysis_name='renamed')


def test_a_model_of_the_wrong_width_is_rejected(dataset, tmp_path):
    """The last way the three can disagree, once the digests agree.

    The label list, its digest and ``n_train_stimuli`` are all made consistent
    here, so what fires is the check at the contraction rather than an earlier
    one -- and its message names the invariant, not a digest.
    """
    decoder_dir = str(tmp_path / 'decoders')
    _train(dataset, decoder_dir)
    model_dir = pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC')

    labels_path = os.path.join(model_dir, 'train_labels.json')
    with open(labels_path) as f:
        labels = json.load(f)
    truncated = labels[:-1]
    with open(labels_path, 'w') as f:
        json.dump(truncated, f)

    meta_path = os.path.join(model_dir, 'factorized.yaml')
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    meta['labels_digest'] = labels_digest(truncated)
    meta['n_train_stimuli'] = len(truncated)
    with open(meta_path, 'w') as f:
        yaml.dump(meta, f)

    with pytest.raises(ValueError, match='column i must be combined'):
        _predict(dataset, decoder_dir, str(tmp_path / 'decoded'),
                 analysis_name='narrow')


# The row assembly ##############################################################

def test_a_missing_training_stimulus_is_rejected(dataset, tmp_path):
    decoder_dir = str(tmp_path / 'decoders')
    _train(dataset, decoder_dir)

    partial = str(tmp_path / 'partial_features')
    synthetic.write_feature_tree(partial, {LAYER: dataset.layer_shapes[LAYER]},
                                 dataset.unique_train_labels[:-1], seed=3)

    with pytest.raises(ValueError, match='exactly one feature store'):
        _predict(dataset, decoder_dir, str(tmp_path / 'decoded'),
                 features_paths=[partial], analysis_name='missing')


def test_a_duplicated_stimulus_cannot_hide_behind_a_missing_one(dataset,
                                                                tmp_path):
    """Why the row count alone is not a check.

    Two stores: the first holds every stimulus but the last, the second holds
    only the first stimulus.  ``bdpy``'s ``get_multi_features`` emits one row
    for the duplicated label from each store and none for the missing one,
    landing on exactly ``len(labels)`` rows with every row after the duplicate
    shifted against its coefficient column.
    """
    from bdpy.dataform import Features
    from bdpy.dataform.utils import get_multi_features

    decoder_dir = str(tmp_path / 'decoders')
    _train(dataset, decoder_dir)

    labels = dataset.unique_train_labels
    shapes = {LAYER: dataset.layer_shapes[LAYER]}
    first = str(tmp_path / 'store_a')
    second = str(tmp_path / 'store_b')
    synthetic.write_feature_tree(first, shapes, labels[:-1], seed=4)
    synthetic.write_feature_tree(second, shapes, labels[:1], seed=5)

    stores = [Features(first), Features(second)]
    assert get_multi_features(stores, LAYER, labels=labels).shape[0] == \
        len(labels), 'the bdpy behavior this test is about has changed'

    loader = TrainingFeatureLoader([first, second])
    with pytest.raises(ValueError, match='exactly one feature store'):
        loader.get(LAYER, labels)

    with pytest.raises(ValueError, match='exactly one feature store'):
        _predict(dataset, decoder_dir, str(tmp_path / 'decoded'),
                 features_paths=[first, second], analysis_name='duplicated')


def test_a_multi_row_feature_file_is_rejected(dataset, tmp_path):
    """One row per stimulus, or the rows after it are shifted."""
    from bdpy.dataform import save_array

    labels = dataset.unique_train_labels
    store = str(tmp_path / 'store')
    synthetic.write_feature_tree(store, {LAYER: (4,)}, labels, seed=6)
    save_array(os.path.join(store, LAYER, '%s.mat' % labels[0]),
               np.zeros((2, 4), dtype=np.float32), key='feat',
               dtype=np.float32, sparse=False)

    loader = TrainingFeatureLoader([store])
    with pytest.raises(ValueError, match='contributes 2 feature rows'):
        loader.get(LAYER, labels)


def test_the_loader_returns_one_row_per_label_in_order(dataset, tmp_path):
    """Two stores that partition the stimuli: still one row each, in order."""
    labels = dataset.unique_train_labels
    shapes = {LAYER: dataset.layer_shapes[LAYER]}
    first = str(tmp_path / 'store_a')
    second = str(tmp_path / 'store_b')
    left = synthetic.write_feature_tree(first, shapes, labels[:2], seed=8)
    right = synthetic.write_feature_tree(second, shapes, labels[2:], seed=9)

    loader = TrainingFeatureLoader([first, second])
    requested = list(reversed(labels))
    features = loader.get(LAYER, requested)

    by_label = dict(zip(labels[:2], left[LAYER]))
    by_label.update(zip(labels[2:], right[LAYER]))
    np.testing.assert_array_equal(
        features, np.vstack([by_label[lb][np.newaxis] for lb in requested]))


# The feature index is prediction-side selection ################################

def test_index_path_resolves_inside_the_feature_directory():
    """Not against the current directory, which is where the old copy looked."""
    assert resolve_feature_index_path(
        os.path.join('data', 'features'), 'index_random1000.mat') == os.path.join(
            'data', 'features', 'index_random1000.mat')


def test_absolute_index_path_is_left_alone():
    absolute = os.path.join(os.sep, 'somewhere', 'index.mat')
    assert resolve_feature_index_path('data/features', absolute) == absolute


def test_the_feature_index_selects_units_at_prediction_time(dataset, tmp_path):
    """The decoder knows nothing about it, and does not need to.

    Training never reads a feature file, so an index is not part of the
    decoder: it selects which units of ``F`` prediction combines, and the
    decoded features come out with the selected width.
    """
    decoder_dir = str(tmp_path / 'decoders')
    decoded_dir = str(tmp_path / 'decoded')
    index = [3, 0]
    synthetic.write_feature_index(
        os.path.join(dataset.train_features_dir, 'index_two.mat'),
        {LAYER: index})

    _train(dataset, decoder_dir)
    assert not os.path.exists(os.path.join(decoder_dir, 'feature_index.mat'))

    _predict(dataset, decoder_dir, decoded_dir,
             feature_index_file='index_two.mat', analysis_name='indexed')

    # The index travels with the decoded features, as in the fastl2lir path.
    assert os.path.isfile(os.path.join(decoded_dir, 'feature_index.mat'))

    _, test_labels = legacy_ridge.average_test_brain(
        dataset.brain('sub-01', 'VC', 'test'), dataset.labels('sub-01', 'test'))
    predicted = pipeline.read_decoded_features(decoded_dir, LAYER, 'sub-01',
                                               'VC', test_labels)
    assert predicted.shape == (len(test_labels), len(index))

    train_labels = ridge_factorization.load_train_labels(
        pipeline.brain_model_dir(decoder_dir, 'sub-01', 'VC'))
    selected = dataset.feature_matrix(LAYER, train_labels)[:, index]
    expected = _stimulus_coefficients(dataset, decoder_dir) @ selected
    np.testing.assert_allclose(predicted, expected.astype(np.float32),
                               rtol=1e-5, atol=1e-6)
