"""Reference implementation of the *direct* sklearn Ridge feature decoder.

This mirrors, in plain numpy + scikit-learn, exactly what
``train_decoder_sklearn_ridge.py`` + ``predict_feature.py`` did before the
decoder was factorized, including every detail bdpy's
``ModelTraining``/``ModelTest`` contribute:

* brain z-scoring with ``ddof=1`` followed by ``X[np.isinf(X)] = 0``;
* feature z-scoring computed over the **unique** stimuli;
* one Ridge per index along ``chunk_axis`` when the feature array is at least
  3-D (``Y.ndim >= chunk_ndim + 1`` with bdpy's default ``chunk_ndim=2``);
* the chunk taken with ``np.take(..., [i], axis=chunk_axis)`` so the axis is
  kept with size 1, and the normalization parameters sliced the same way;
* ``Y = Y[feat_index]`` applied *after* normalization -- this is the repetition
  expansion that turns M unique feature rows into N trial-aligned rows;
* ``reshape(n, -1, order='F')`` before fitting and the inverse reshape after
  predicting;
* ``float32`` casts for both X and Y;
* ``np.concatenate(..., axis=chunk_axis)`` to reassemble the chunks;
* un-normalization as ``y_pred * y_norm + y_mean``.

It exists so that the factorized implementation can be compared against the old
math directly, and so that the old math itself is pinned by tests against the
real scripts (see ``tests/test_legacy_reference.py``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import copy

import numpy as np
from sklearn.linear_model import Ridge

CHUNK_NDIM = 2  # bdpy ModelTraining default

# What the *direct* implementation stored per decoder. The factorized decoder
# keeps only the brain half (`pipeline.BRAIN_NORM_KEYS`); the feature half is a
# property of the features a prediction combines and lives next to the decoded
# features.
NORM_KEYS = ('x_mean', 'x_norm', 'y_mean', 'y_norm')


def brain_norm_params(brain: np.ndarray):
    """``(mean, std)`` of the brain data, shaped ``(1, n_voxels)``."""
    mean = np.mean(brain, axis=0)[np.newaxis, :]
    std = np.std(brain, axis=0, ddof=1)[np.newaxis, :]
    return mean, std


def feature_norm_params(feat: np.ndarray):
    """``(mean, std)`` of the features, shaped ``(1, *feature_shape)``."""
    mean = np.mean(feat, axis=0)[np.newaxis, :]
    std = np.std(feat, axis=0, ddof=1)[np.newaxis, :]
    return mean, std


def assignment_index(brain_labels: Sequence[str],
                     feat_labels: Sequence[str]) -> np.ndarray:
    """Row of ``feat`` matching each brain trial (the ``Y_sort`` index)."""
    return np.array([np.where(np.array(feat_labels) == bl)
                     for bl in brain_labels]).flatten()


def normalize_brain_for_training(brain: np.ndarray, mean: np.ndarray,
                                 std: np.ndarray) -> np.ndarray:
    """z-score as ``ModelTraining`` does it, including the ``inf`` clean-up."""
    with np.errstate(invalid='ignore', divide='ignore'):
        x = (brain - mean) / std
    x[np.isinf(x)] = 0
    return x


def use_chunking(feat: np.ndarray, chunk_axis: Optional[int]) -> bool:
    if chunk_axis is None:
        return False
    return feat.ndim >= CHUNK_NDIM + 1


def legacy_train(brain: np.ndarray, brain_labels: Sequence[str],
                 feat: np.ndarray, feat_labels: Sequence[str],
                 alpha: float = 100, chunk_axis: Optional[int] = 1,
                 dtype: Any = np.float32) -> Dict[str, Any]:
    """Fit the direct brain -> feature Ridge decoder, chunk by chunk."""
    x_mean, x_norm = brain_norm_params(brain)
    y_mean, y_norm = feature_norm_params(feat)
    feat_index = assignment_index(brain_labels, feat_labels)

    x = normalize_brain_for_training(brain, x_mean, x_norm)

    chunking = use_chunking(feat, chunk_axis)
    chunk_index: Sequence[Optional[int]]
    chunk_index = range(feat.shape[chunk_axis]) if chunking else [None]

    models: List[Dict[str, Any]] = []
    for i_chunk in chunk_index:
        if chunking:
            y = np.take(feat, [i_chunk], axis=chunk_axis)
            chunk_mean = np.take(y_mean, [i_chunk], axis=chunk_axis)
            chunk_norm = np.take(y_norm, [i_chunk], axis=chunk_axis)
        else:
            y = feat
            chunk_mean, chunk_norm = y_mean, y_norm

        with np.errstate(invalid='ignore', divide='ignore'):
            y = (y - chunk_mean) / chunk_norm
        y[np.isinf(y)] = 0
        y = y[feat_index]

        y_shape = y.shape[1:]
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1, order='F')

        model = Ridge(alpha=alpha)
        model.fit(x.astype(dtype), y.astype(dtype))
        models.append({'model': model, 'y_shape': y_shape})

    return {
        'models': models,
        'x_mean': x_mean, 'x_norm': x_norm,
        'y_mean': y_mean, 'y_norm': y_norm,
        'feat_index': feat_index,
        'chunk_axis': chunk_axis,
        'chunking': chunking,
    }


def legacy_predict(trained: Dict[str, Any], brain_test: np.ndarray,
                   dtype: Any = np.float32) -> np.ndarray:
    """Predict features from *raw* test brain data, as ``predict_feature`` did.

    Note the deliberate asymmetry with training: the prediction script does not
    apply the ``inf`` clean-up after z-scoring.
    """
    x = (brain_test - trained['x_mean']) / trained['x_norm']
    x = x.astype(dtype)

    preds = []
    for entry in trained['models']:
        y_pred = entry['model'].predict(x)
        if y_pred.shape[1:] != entry['y_shape']:
            y_pred = y_pred.reshape((y_pred.shape[0],) + entry['y_shape'],
                                    order='F')
        preds.append(y_pred)

    if trained['chunk_axis'] is None:
        pred = preds[0]
    else:
        pred = np.concatenate(preds, axis=trained['chunk_axis'])

    return pred * trained['y_norm'] + trained['y_mean']


def average_test_brain(brain: np.ndarray, labels: Sequence[str],
                       excluded_labels: Sequence[str] = ()):
    """Trial-average the test brain data, as ``predict_feature`` does."""
    unique = [lb for lb in np.unique(labels) if lb not in excluded_labels]
    averaged = np.vstack([
        np.mean(brain[(np.array(labels) == lb).flatten(), :], axis=0)
        for lb in unique])
    return averaged, list(unique)


def single_trial_labels(labels: Sequence[str]) -> List[str]:
    """Output labels used by ``predict_feature`` when ``average_sample`` is off."""
    return ['sample{:06}-{}'.format(i + 1, lb) for i, lb in enumerate(labels)]


def clone_trained(trained: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(trained)


def write_legacy_decoder(model_dir: str, trained: Dict[str, Any]) -> None:
    """Write ``trained`` in the pre-factorization on-disk decoder format.

    One pickle per chunk (``%08d.pkl.gz``, or ``model.pkl.gz`` when the target
    was not chunked) holding ``{'model': ..., 'y_shape': ...}``, the four
    normalization ``.mat`` files, and an ``info.yaml`` completion marker.  Used
    to test that the current prediction script can still read decoders produced
    by the direct implementation.
    """
    import os
    import pickle

    import yaml
    from bdpy.dataform import save_array

    os.makedirs(model_dir, exist_ok=True)

    for key in ('x_mean', 'x_norm', 'y_mean', 'y_norm'):
        save_array(os.path.join(model_dir, key + '.mat'), trained[key],
                   key=key, dtype=np.float32, sparse=False)

    for i, entry in enumerate(trained['models']):
        name = ('%08d.pkl.gz' % i) if trained['chunking'] else 'model.pkl.gz'
        with open(os.path.join(model_dir, name), 'wb') as f:
            pickle.dump({'model': entry['model'], 'y_shape': entry['y_shape']},
                        f, protocol=4)

    with open(os.path.join(model_dir, 'info.yaml'), 'w') as f:
        f.write(yaml.dump({'_status': {'computation_id': 'legacy',
                                       'computation_status': 'done'}},
                          default_flow_style=False))
