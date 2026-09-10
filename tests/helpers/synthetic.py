"""Build tiny synthetic on-disk datasets for the feature-decoding tests.

Everything is small enough to live in a temporary directory: a handful of
``bdpy.BData`` HDF5 files and a ``bdpy.dataform.Features`` tree.  No part of the
real DeepRecon dataset is needed.

The dataset intentionally reproduces the properties the decoders depend on:

* stimuli are **repeated** across fMRI trials, with an *unbalanced* number of
  repetitions per stimulus (3, 3, 2, 2, 1, 1);
* one subject's training trials are split over **two** ``.h5`` files, so
  ``select_data_multi_bdatas`` / ``get_labels_multi_bdatas`` are exercised;
* two ROIs of different sizes;
* a 1-D ``fc``-like feature layer (no chunking) *and* a 3-D ``conv``-like layer,
  so the ``chunk_axis`` / ``order='F'`` reshape path is exercised.  The shipped
  config only enables ``fc6``/``fc7``/``fc8``, which never reaches that path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple
import os

import numpy as np
from bdpy import BData
from bdpy.dataform import save_array

# Feature layers: name -> feature shape of a single stimulus.
LAYER_SHAPES: Dict[str, Tuple[int, ...]] = {
    'conv_like': (3, 2, 2),
    'fc_like': (4,),
}

N_VOXELS = 8
ROIS: Dict[str, str] = {'VC': 'ROI_VC = 1', 'HALF': 'ROI_HALF = 1'}
ROI_SIZES: Dict[str, int] = {'VC': N_VOXELS, 'HALF': N_VOXELS // 2}
LABEL_KEY = 'stimulus_name'

# Unbalanced repetitions: 6 unique training stimuli, 12 trials.
TRAIN_REPEATS: Tuple[int, ...] = (3, 3, 2, 2, 1, 1)
# 3 unique test stimuli, 5 trials.
TEST_REPEATS: Tuple[int, ...] = (2, 2, 1)


@dataclass
class SyntheticDataset:
    """Paths and in-memory copies of a synthetic decoding dataset."""

    root: str
    train_fmri: Dict[str, List[str]]
    test_fmri: Dict[str, List[str]]
    train_features_dir: str
    train_brain: Dict[str, np.ndarray] = field(default_factory=dict)
    train_labels: Dict[str, List[str]] = field(default_factory=dict)
    test_brain: Dict[str, np.ndarray] = field(default_factory=dict)
    test_labels: Dict[str, List[str]] = field(default_factory=dict)
    features: Dict[str, np.ndarray] = field(default_factory=dict)
    unique_train_labels: List[str] = field(default_factory=list)
    unique_test_labels: List[str] = field(default_factory=list)
    rois: Dict[str, str] = field(default_factory=lambda: dict(ROIS))
    label_key: str = LABEL_KEY
    layer_shapes: Dict[str, Tuple[int, ...]] = field(
        default_factory=lambda: dict(LAYER_SHAPES))

    @property
    def layers(self) -> List[str]:
        return sorted(self.layer_shapes)

    def brain(self, subject: str, roi: str, split: str = 'train') -> np.ndarray:
        """Voxel data for one subject/ROI, as the scripts would select it."""
        data = self.train_brain if split == 'train' else self.test_brain
        return data[subject][:, :ROI_SIZES[roi]]

    def labels(self, subject: str, split: str = 'train') -> List[str]:
        return (self.train_labels if split == 'train' else self.test_labels)[subject]

    def feature_matrix(self, layer: str, labels: Sequence[str]) -> np.ndarray:
        """Rows of ``features[layer]`` in the order of ``labels``."""
        index = [self.unique_train_labels.index(lb) for lb in labels]
        return self.features[layer][index]


def _expand(labels: Sequence[str], repeats: Sequence[int]) -> List[str]:
    """Repeat each label ``repeats[i]`` times, interleaved as in a real run."""
    trials: List[str] = []
    remaining = list(repeats)
    while any(r > 0 for r in remaining):
        for i, label in enumerate(labels):
            if remaining[i] > 0:
                trials.append(label)
                remaining[i] -= 1
    return trials


def write_bdata(path: str, brain: np.ndarray, labels: Sequence[str],
                label_to_number: Dict[str, int]) -> None:
    """Write one ``BData`` HDF5 file with ROI metadata and a label vmap."""
    n_voxels = brain.shape[1]
    bdata = BData()
    bdata.add(np.asarray(brain, dtype=float), 'VoxelData')
    bdata.add(np.array([[label_to_number[lb]] for lb in labels], dtype=float),
              LABEL_KEY)
    bdata.add_metadata('ROI_VC', np.ones(n_voxels), where='VoxelData')
    half = np.zeros(n_voxels)
    half[:ROI_SIZES['HALF']] = 1
    bdata.add_metadata('ROI_HALF', half, where='VoxelData')
    bdata.add_vmap(LABEL_KEY, {v: k for k, v in label_to_number.items()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bdata.save(path)


def write_features(root: str, layer_shapes: Dict[str, Tuple[int, ...]],
                   labels: Sequence[str],
                   arrays: Dict[str, np.ndarray]) -> None:
    """Write a ``Features`` tree ``<root>/<layer>/<label>.mat`` (key ``feat``)."""
    for layer in layer_shapes:
        layer_dir = os.path.join(root, layer)
        os.makedirs(layer_dir, exist_ok=True)
        for i, label in enumerate(labels):
            save_array(os.path.join(layer_dir, label + '.mat'),
                       arrays[layer][i][np.newaxis],
                       key='feat', dtype=np.float32, sparse=False)


def make_dataset(root: str, seed: int = 0,
                 constant_feature_unit: bool = False) -> SyntheticDataset:
    """Create a complete synthetic dataset under ``root``.

    Parameters
    ----------
    root
        Directory to create the dataset in.
    seed
        Seed for the random voxel/feature values.
    constant_feature_unit
        If True, one feature unit of every layer is constant across the training
        stimuli, i.e. it has zero training variance.  Used to pin the behavior of
        the decoder for such units.
    """
    rng = np.random.RandomState(seed)

    n_train = len(TRAIN_REPEATS)
    n_test = len(TEST_REPEATS)
    unique_train_labels = ['train%04d' % (i + 1) for i in range(n_train)]
    unique_test_labels = ['test%04d' % (i + 1) for i in range(n_test)]
    label_to_number = {lb: i + 1 for i, lb in
                       enumerate(unique_train_labels + unique_test_labels)}

    train_trial_labels = _expand(unique_train_labels, TRAIN_REPEATS)
    test_trial_labels = _expand(unique_test_labels, TEST_REPEATS)

    # Features of the unique training stimuli.
    features: Dict[str, np.ndarray] = {}
    for layer, shape in LAYER_SHAPES.items():
        feat = rng.randn(n_train, *shape).astype(np.float32) * 2.0 + 1.0
        if constant_feature_unit:
            flat = feat.reshape(n_train, -1)
            flat[:, 0] = 7.0
            feat = flat.reshape((n_train,) + shape)
        features[layer] = feat

    # Brain data, weakly coupled to the features so the decoder has signal.
    train_brain: Dict[str, np.ndarray] = {}
    train_labels: Dict[str, List[str]] = {}
    test_brain: Dict[str, np.ndarray] = {}
    test_labels: Dict[str, List[str]] = {}

    driver = features['fc_like'].reshape(n_train, -1)
    mixing = rng.randn(driver.shape[1], N_VOXELS)

    for subject in ('sub-01', 'sub-02'):
        idx = [unique_train_labels.index(lb) for lb in train_trial_labels]
        signal = driver[idx] @ mixing
        train_brain[subject] = (
            signal + rng.randn(len(train_trial_labels), N_VOXELS) * 0.5)
        train_labels[subject] = list(train_trial_labels)
        test_brain[subject] = rng.randn(len(test_trial_labels), N_VOXELS) * 3.0
        test_labels[subject] = list(test_trial_labels)

    # sub-01's training trials are split over two files, sub-02's over one.
    train_fmri: Dict[str, List[str]] = {}
    split = 7
    sub01 = [os.path.join(root, 'fmri', 'sub-01_train_run1.h5'),
             os.path.join(root, 'fmri', 'sub-01_train_run2.h5')]
    write_bdata(sub01[0], train_brain['sub-01'][:split],
                train_labels['sub-01'][:split], label_to_number)
    write_bdata(sub01[1], train_brain['sub-01'][split:],
                train_labels['sub-01'][split:], label_to_number)
    train_fmri['sub-01'] = sub01

    sub02 = [os.path.join(root, 'fmri', 'sub-02_train.h5')]
    write_bdata(sub02[0], train_brain['sub-02'], train_labels['sub-02'],
                label_to_number)
    train_fmri['sub-02'] = sub02

    # ``predict_feature.py`` reads only the first file of each subject.
    test_fmri: Dict[str, List[str]] = {}
    for subject in ('sub-01', 'sub-02'):
        path = os.path.join(root, 'fmri', '%s_test.h5' % subject)
        write_bdata(path, test_brain[subject], test_labels[subject],
                    label_to_number)
        test_fmri[subject] = [path]

    train_features_dir = os.path.join(root, 'features', 'train')
    write_features(train_features_dir, LAYER_SHAPES, unique_train_labels,
                   features)

    return SyntheticDataset(
        root=root,
        train_fmri=train_fmri,
        test_fmri=test_fmri,
        train_features_dir=train_features_dir,
        train_brain=train_brain,
        train_labels=train_labels,
        test_brain=test_brain,
        test_labels=test_labels,
        features=features,
        unique_train_labels=unique_train_labels,
        unique_test_labels=unique_test_labels,
    )


def add_layer(dataset: SyntheticDataset, name: str, shape: Tuple[int, ...],
              seed: int = 1) -> np.ndarray:
    """Add one more feature layer to an existing dataset's feature tree."""
    rng = np.random.RandomState(seed)
    n_stimuli = len(dataset.unique_train_labels)
    array = rng.randn(n_stimuli, *shape).astype(np.float32)
    write_features(dataset.train_features_dir, {name: shape},
                   dataset.unique_train_labels, {name: array})
    dataset.layer_shapes[name] = shape
    dataset.features[name] = array
    return array


def write_test_features(dataset: SyntheticDataset, seed: int = 21) -> str:
    """Write a ``Features`` tree for the *test* stimuli and return its path.

    ``evaluation.py`` compares the decoded features against the true features
    of the test stimuli, which the training dataset does not contain.
    """
    root = os.path.join(dataset.root, 'features', 'test')
    write_feature_tree(root, dataset.layer_shapes, dataset.unique_test_labels,
                       seed=seed)
    return root


def write_feature_index(path: str,
                        index_by_layer: Dict[str, Sequence[int]]) -> None:
    """Write a feature-index ``.mat`` as ``Features(feature_index=...)`` expects.

    bdpy reads it as ``hdf5storage.loadmat(path)['index']`` and then indexes the
    result by layer name, so ``index`` is a mapping from layer to a 1-D array of
    zero-based column indices into the C-order-flattened features.
    """
    import hdf5storage

    payload = {layer: np.asarray(index).ravel()
               for layer, index in index_by_layer.items()}
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    hdf5storage.savemat(path, {'index': payload}, format='7.3',
                        oned_as='column', store_python_metadata=True)


def write_feature_tree(root: str, layer_shapes: Dict[str, Tuple[int, ...]],
                       labels: Sequence[str], seed: int = 0
                       ) -> Dict[str, np.ndarray]:
    """Write an independent ``Features`` tree and return its arrays."""
    rng = np.random.RandomState(seed)
    arrays = {layer: rng.randn(len(labels), *shape).astype(np.float32)
              for layer, shape in layer_shapes.items()}
    write_features(root, layer_shapes, labels, arrays)
    return arrays
