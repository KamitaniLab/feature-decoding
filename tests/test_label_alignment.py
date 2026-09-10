"""Alignment between fMRI trials and unique-stimulus features.

The training script derives ``feat_labels = np.unique(brain_labels)``, loads one
feature row per unique stimulus, and expands those rows back to trial order with
``feat_index`` (bdpy's ``Y_sort``).  That expansion is exactly the one-hot
assignment ``Y = R F`` the factorized decoder relies on, so it is pinned here.
"""

from __future__ import annotations

import numpy as np

from tests.helpers import legacy_ridge, pipeline


def test_assignment_index_maps_every_trial_to_its_stimulus(dataset):
    brain_labels = dataset.labels('sub-01', 'train')
    feat_labels = list(np.unique(brain_labels))

    index = legacy_ridge.assignment_index(brain_labels, feat_labels)

    assert index.shape == (len(brain_labels),)
    assert [feat_labels[i] for i in index] == list(brain_labels)


def test_unique_labels_are_sorted_and_deduplicated(dataset):
    brain_labels = dataset.labels('sub-01', 'train')
    feat_labels = list(np.unique(brain_labels))

    assert feat_labels == sorted(set(brain_labels))
    assert len(feat_labels) < len(brain_labels)


def test_repetitions_are_unbalanced_in_the_fixture(dataset):
    """The fixture must keep unbalanced repeats, or several tests go blind."""
    brain_labels = dataset.labels('sub-01', 'train')
    counts = [brain_labels.count(lb) for lb in np.unique(brain_labels)]

    assert len(set(counts)) > 1, counts


def test_expansion_repeats_feature_rows_per_trial(dataset):
    brain_labels = dataset.labels('sub-01', 'train')
    feat_labels = list(np.unique(brain_labels))
    feat = dataset.feature_matrix('conv_like', feat_labels)

    index = legacy_ridge.assignment_index(brain_labels, feat_labels)
    expanded = feat[index]

    assert expanded.shape == (len(brain_labels),) + feat.shape[1:]
    for trial, label in enumerate(brain_labels):
        np.testing.assert_array_equal(
            expanded[trial], feat[feat_labels.index(label)])


def test_trial_order_does_not_matter_for_the_fitted_decoder(dataset,
                                                            tmp_path):
    """Permuting (brain trial, label) pairs together leaves predictions intact."""
    from tests.helpers import synthetic

    layer, subject, roi = 'fc_like', 'sub-01', 'VC'
    brain = dataset.brain(subject, roi, 'train')
    brain_labels = dataset.labels(subject, 'train')
    feat_labels = list(np.unique(brain_labels))
    feat = dataset.feature_matrix(layer, feat_labels)

    trained = legacy_ridge.legacy_train(brain, brain_labels, feat, feat_labels,
                                        alpha=100, chunk_axis=1)

    order = np.random.RandomState(1).permutation(len(brain_labels))
    shuffled = legacy_ridge.legacy_train(
        brain[order], [brain_labels[i] for i in order], feat, feat_labels,
        alpha=100, chunk_axis=1)

    test_brain, _ = legacy_ridge.average_test_brain(
        dataset.brain(subject, roi, 'test'), dataset.labels(subject, 'test'))
    np.testing.assert_allclose(
        legacy_ridge.legacy_predict(trained, test_brain),
        legacy_ridge.legacy_predict(shuffled, test_brain),
        rtol=1e-4, atol=1e-5)
