"""Exact equivalence of the direct and factorized Ridge formulations.

These tests work in ``float64`` on small arrays, so the tolerances can be tight
enough to be a real statement about the algebra rather than about float32
rounding.  See ``ridge_factorization`` for the derivation; the identities
checked here are the three steps it rests on:

1. ``Ridge(alpha).fit(X, R @ F).predict(Xt) == Ridge(alpha).fit(X, R).predict(Xt) @ F``
2. the rows of the predicted stimulus coefficients sum to one, and
3. therefore the feature mean/SD normalization cancels.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from ridge_factorization import combine_features, one_hot_assignment

RTOL = 1e-9
ATOL = 1e-9


def make_problem(n_stimuli=6, repeats=(3, 3, 2, 2, 1, 1), n_voxels=5,
                 n_features=7, n_test=4, seed=0):
    """Random brain/feature problem with repeated stimulus presentations."""
    assert len(repeats) == n_stimuli
    rng = np.random.RandomState(seed)

    labels = []
    for stimulus, count in enumerate(repeats):
        labels.extend([stimulus] * count)
    unique_labels = list(range(n_stimuli))

    assignment = one_hot_assignment(labels, unique_labels)
    features = rng.randn(n_stimuli, n_features)
    brain = rng.randn(len(labels), n_voxels)
    brain_test = rng.randn(n_test, n_voxels)
    return brain, brain_test, assignment, features


def direct_prediction(brain, brain_test, assignment, features, alpha):
    """Fit on the expanded target ``Y = R F`` -- the original formulation."""
    model = Ridge(alpha=alpha)
    model.fit(brain, assignment @ features)
    return model.predict(brain_test)


def factorized_prediction(brain, brain_test, assignment, features, alpha,
                          chunk_axis=None):
    """Fit on the stimulus basis ``R``, then combine with ``F``."""
    model = Ridge(alpha=alpha)
    model.fit(brain, assignment)
    coefficients = model.predict(brain_test)
    return combine_features(coefficients, features, chunk_axis=chunk_axis)


@pytest.mark.parametrize('alpha', [0.0, 1e-3, 1.0, 100.0, 1e6])
def test_equivalent_for_a_range_of_alpha(alpha):
    brain, brain_test, assignment, features = make_problem()

    np.testing.assert_allclose(
        factorized_prediction(brain, brain_test, assignment, features, alpha),
        direct_prediction(brain, brain_test, assignment, features, alpha),
        rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('repeats', [
    (5, 5, 5, 5, 5, 5),      # balanced repetitions
    (3, 3, 2, 2, 1, 1),      # unbalanced repetitions
    (1, 1, 1, 1, 1, 1),      # no repetition at all (M == N)
    (10, 1, 1, 1, 1, 1),     # strongly unbalanced
])
def test_equivalent_for_any_repetition_structure(repeats):
    brain, brain_test, assignment, features = make_problem(repeats=repeats)

    np.testing.assert_allclose(
        factorized_prediction(brain, brain_test, assignment, features, 100.0),
        direct_prediction(brain, brain_test, assignment, features, 100.0),
        rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('n_features', [1, 2, 50])
def test_equivalent_for_single_and_multi_output(n_features):
    brain, brain_test, assignment, features = make_problem(
        n_features=n_features)

    actual = factorized_prediction(brain, brain_test, assignment, features,
                                   100.0)
    expected = direct_prediction(brain, brain_test, assignment, features,
                                 100.0)

    # scikit-learn ravels a single-column target, so for n_features == 1 the
    # direct formulation returns shape (n_test,) where the factorized one keeps
    # (n_test, 1).  The values are what the equivalence is about; the real
    # pipeline always has n_features >= 2 anyway.
    np.testing.assert_allclose(actual, expected.reshape(actual.shape),
                               rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('n_voxels,n_stimuli,repeats', [
    (3, 6, (3, 3, 2, 2, 1, 1)),     # fewer voxels than stimuli
    (40, 6, (3, 3, 2, 2, 1, 1)),    # more voxels than trials (dual solver)
])
def test_equivalent_regardless_of_problem_shape(n_voxels, n_stimuli, repeats):
    brain, brain_test, assignment, features = make_problem(
        n_stimuli=n_stimuli, repeats=repeats, n_voxels=n_voxels)

    np.testing.assert_allclose(
        factorized_prediction(brain, brain_test, assignment, features, 100.0),
        direct_prediction(brain, brain_test, assignment, features, 100.0),
        rtol=RTOL, atol=ATOL)


def test_equivalent_for_multidimensional_features():
    brain, brain_test, assignment, _ = make_problem()
    rng = np.random.RandomState(3)
    features = rng.randn(assignment.shape[1], 3, 2, 2)

    flat = features.reshape(features.shape[0], -1)
    expected = direct_prediction(brain, brain_test, assignment, flat, 100.0)
    expected = expected.reshape((brain_test.shape[0],) + features.shape[1:])

    for chunk_axis in (None, 1, 2, 3):
        actual = factorized_prediction(brain, brain_test, assignment, features,
                                       100.0, chunk_axis=chunk_axis)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def test_predicted_stimulus_coefficients_sum_to_one():
    """``C @ 1 == 1``: the identity that makes the feature mean cancel."""
    brain, brain_test, assignment, _ = make_problem()

    for alpha in (1e-3, 1.0, 100.0, 1e6):
        model = Ridge(alpha=alpha).fit(brain, assignment)
        coefficients = model.predict(brain_test)
        np.testing.assert_allclose(coefficients.sum(axis=1),
                                   np.ones(brain_test.shape[0]),
                                   rtol=RTOL, atol=ATOL)


def test_feature_normalization_cancels():
    """Normalizing the target and un-normalizing the prediction is a no-op.

    The direct pipeline fitted on ``(R F - m) / s`` and un-normalized with
    ``* s + m``.  Because the stimulus coefficients sum to one, that round trip
    returns exactly the un-normalized linear combination.
    """
    brain, brain_test, assignment, features = make_problem()

    mean = np.mean(features, axis=0)[np.newaxis, :]
    std = np.std(features, axis=0, ddof=1)[np.newaxis, :]
    assert np.all(std > 0)

    normalized = direct_prediction(brain, brain_test, assignment,
                                   (features - mean) / std, 100.0)
    round_tripped = normalized * std + mean

    np.testing.assert_allclose(
        factorized_prediction(brain, brain_test, assignment, features, 100.0),
        round_tripped, rtol=RTOL, atol=ATOL)


def test_intercept_is_still_fitted_and_applied():
    """The factorization keeps sklearn's intercept, it does not drop it."""
    brain, brain_test, assignment, features = make_problem()

    model = Ridge(alpha=100.0).fit(brain, assignment)
    assert np.any(np.abs(model.intercept_) > 1e-6)

    with_intercept = model.predict(brain_test)
    without_intercept = brain_test @ model.coef_.T
    assert not np.allclose(with_intercept, without_intercept)


def test_intercept_is_what_makes_the_feature_normalization_cancel():
    """Locate precisely what ``fit_intercept=True`` (the default) buys.

    The factorization itself does not need the intercept: Ridge prediction is
    linear in the target either way, so ``S(R F) == (S R) F`` holds with
    ``fit_intercept=False`` too.  What the intercept buys is ``C @ 1 == 1``, and
    *that* is what makes the feature mean/SD round trip cancel.  Since the
    implementation drops that round trip, it does depend on the default.
    """
    brain, brain_test, assignment, features = make_problem()

    direct = Ridge(alpha=100.0, fit_intercept=False)
    direct.fit(brain, assignment @ features)
    basis = Ridge(alpha=100.0, fit_intercept=False)
    basis.fit(brain, assignment)
    coefficients = basis.predict(brain_test)

    # The two formulations still agree without an intercept ...
    np.testing.assert_allclose(combine_features(coefficients, features),
                               direct.predict(brain_test),
                               rtol=RTOL, atol=ATOL)

    # ... but the rows no longer sum to one, so the un-normalization would not
    # cancel any more.
    assert not np.allclose(coefficients.sum(axis=1), 1.0)

    mean = np.mean(features, axis=0)[np.newaxis, :]
    std = np.std(features, axis=0, ddof=1)[np.newaxis, :]
    normalized = Ridge(alpha=100.0, fit_intercept=False).fit(
        brain, assignment @ ((features - mean) / std)).predict(brain_test)
    assert not np.allclose(normalized * std + mean,
                           combine_features(coefficients, features),
                           rtol=1e-6)


def test_one_hot_assignment_matches_the_expansion():
    labels = ['b', 'a', 'b', 'c', 'a', 'b']
    unique_labels = sorted(set(labels))

    assignment = one_hot_assignment(labels, unique_labels)

    assert assignment.shape == (len(labels), len(unique_labels))
    np.testing.assert_array_equal(assignment.sum(axis=1), np.ones(len(labels)))

    rng = np.random.RandomState(0)
    features = rng.randn(len(unique_labels), 4)
    expected = features[[unique_labels.index(lb) for lb in labels]]
    np.testing.assert_array_equal(assignment @ features, expected)


def test_one_hot_assignment_rejects_unknown_labels():
    with pytest.raises(ValueError, match='not in unique_labels'):
        one_hot_assignment(['a', 'z'], ['a', 'b'])


def test_one_hot_assignment_rejects_duplicate_basis_labels():
    with pytest.raises(ValueError, match='duplicates'):
        one_hot_assignment(['a'], ['a', 'a'])
