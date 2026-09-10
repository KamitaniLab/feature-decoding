'''Factorized formulation of the sklearn Ridge feature decoder.

Motivation
----------

The direct decoder regresses brain activity on DNN features and stores a
coefficient matrix of shape ``(d_out, d_in)``.  For DeepRecon-scale features
that matrix is enormous (``d_out`` reaches ~10^7 per VGG-19 layer against
``d_in`` ~10^4 voxels), even though the training targets live in a space of at
most ``M`` dimensions: every stimulus is presented several times, so the target
matrix is ``M`` unique feature vectors repeated over ``N`` trials.

Notation
--------

``X`` (N x p)
    Raw brain data, one row per fMRI trial, with labels ``l_1 ... l_N``.
``L_1 ... L_M``
    The unique stimulus labels, ``np.unique(brain_labels)``.
``R`` (N x M)
    One-hot assignment, ``R[i, j] = 1`` iff ``l_i == L_j``.  Every row sums to
    one, i.e. ``R @ 1 == 1``.
``F`` (M x d)
    Features of the unique stimuli, rows ordered like ``L``.
``m``, ``s`` (1 x d)
    Feature mean / SD over the ``M`` unique rows (``ddof=1``).

The direct decoder fits ``Ridge(alpha)`` on ``X_n = (X - x_mean) / x_norm`` and
``Y_n = (R F - m) / s``, then un-normalizes the prediction as ``* s + m``.  Note
``Y_n = R F_n`` with ``F_n = (F - m) / s``, exactly, because ``R @ 1 == 1``.

Equivalence
-----------

With ``fit_intercept=True`` the sklearn Ridge prediction is affine in the
target.  Writing ``X_c = X_n - 1 x_bar_n`` and

    S = (X_t - 1 x_bar_n) (X_c^T X_c + alpha I)^-1 X_c^T        (n_test x N)

which depends on the brain data only,

    predict(X_t) = S Y + (1 - S 1) y_bar,    y_bar = (1/N) 1^T Y

Substituting ``Y = R F_n`` and ``y_bar = r_bar F_n`` with ``r_bar = (1/N) 1^T R``,

    predict = [ S R + (1 - S 1) r_bar ] F_n = C F_n

and ``C = S R + (1 - S 1) r_bar`` is precisely what the *same* estimator returns
when it is fitted on ``R`` instead:

    C = Ridge(alpha).fit(X_n, R).predict(X_t)                   (n_test x M)

Row sums: ``C 1 = S R 1 + (1 - S 1)(r_bar 1) = S 1 + (1 - S 1) = 1``.  The final
un-normalization therefore collapses:

    (C F_n) * s + m = C (F - 1 m) + m = C F - (C 1) m + m = C F

So the whole decoder is ``Ridge(alpha).fit(X_n, R).predict(X_t) @ F``: the
feature mean/SD normalization cancels exactly, the meaning of ``alpha`` is
unchanged (same ``X``, same penalty), and the intercept is still fitted and
applied -- just on the ``M``-dimensional target.

What is stored shrinks from ``d_out x d_in`` to ``M x d_in`` and no longer
depends on the feature layer, so a single fit per (subject, ROI) serves every
layer.  ``F`` itself is not copied into the decoder: it already exists on disk
as the training feature directory, and prediction loads it from there.

Note also that the fit is on ``R``, which is built from the trial labels alone,
so *training never reads* ``F``: what it learns is the map from brain activity
to coefficients over the training stimuli, plus the identity and order of those
stimuli.  So the artifact records those labels and nothing about the features,
and the one thing it has to guarantee is

    coefficient column i is multiplied by the feature row of training stimulus i

Three cheap checks cover it: ``verify_train_labels`` (the stored label list is
the one the coefficients were fitted with, in order),
``TrainingFeatureLoader.get`` (exactly one feature row from exactly one store
per requested label, assembled in the requested order), and
``check_column_correspondence`` at the contraction.

This is a statement about the *fit*, not a licence to point one decoder
directory at several feature sets: the surrounding pipeline stores one set of
feature statistics per decoder directory for ``evaluation.py``, so one training
feature configuration means one decoder directory.  The layout follows from the
same two facts -- the model does not depend on the layer, the statistics do:

    <decoder>/<subject>/<roi>/model/          the brain-side model, once
    <decoder>/<layer>/<subject>/<roi>/model/  y_mean.mat, y_norm.mat

The statistics are compatibility artifacts for ``evaluation.py``, not
parameters of the model -- prediction never reads them, because the feature
normalization cancels -- and they are written by ``predict_feature.py``, the
step that has the features.
'''


from typing import Any, Dict, List, Optional, Sequence, Tuple

import hashlib
import json
import os
import pickle
import uuid

import numpy as np
import yaml
from bdpy.dataform import Features, save_array
from bdpy.util import makedir_ifnot
from sklearn.linear_model import Ridge


# Model artifact ##############################################################

FORMAT_NAME = 'sklearn_ridge_factorized'
FORMAT_VERSION = 1

MODEL_FILE = 'model.pkl.gz'          # bdpy ModelTraining pickle layout
LABELS_FILE = 'train_labels.json'
META_FILE = 'factorized.yaml'
FEATURE_INDEX_FILE = 'feature_index.mat'

# Identifies how `labels_digest` canonicalizes its input, so the recipe can
# change later without silently comparing incomparable digests.
LABELS_DIGEST_SPEC = 'sha256/labels-v1'

PICKLE_PROTOCOL = 4                  # matches bdpy.ml.ModelTraining

# bdpy ModelTraining/ModelTest chunk the target only when Y.ndim >= chunk_ndim + 1.
CHUNK_NDIM = 2


# Brain-side model ############################################################

def one_hot_assignment(brain_labels: Sequence[Any],
                       unique_labels: Sequence[Any],
                       dtype: Any = np.float64) -> np.ndarray:
    '''Return the assignment matrix ``R`` (n_trials x n_unique_stimuli).

    ``R[i, j] == 1`` iff trial ``i`` presented ``unique_labels[j]``.  This is
    the same alignment bdpy's ``Y_sort`` index expresses; expressed as a matrix
    it makes ``Y == R @ F`` explicit.
    '''
    unique_labels = list(unique_labels)
    position = {label: j for j, label in enumerate(unique_labels)}
    if len(position) != len(unique_labels):
        raise ValueError('unique_labels contains duplicates')

    assignment = np.zeros((len(brain_labels), len(unique_labels)), dtype=dtype)
    for i, label in enumerate(brain_labels):
        try:
            assignment[i, position[label]] = 1
        except KeyError:
            raise ValueError('Label %r of trial %d is not in unique_labels'
                             % (label, i))
    return assignment


def normalize_brain_for_training(brain: np.ndarray, mean: np.ndarray,
                                 std: np.ndarray) -> np.ndarray:
    '''z-score the brain data the way ``bdpy.ml.ModelTraining`` does.

    Infinities produced by zero-variance voxels are replaced by zero, matching
    ``ModelTraining.run()``.  NaNs are deliberately left alone: the original
    implementation does not clean them either.
    '''
    with np.errstate(invalid='ignore', divide='ignore'):
        normalized = (brain - mean) / std
    normalized[np.isinf(normalized)] = 0
    return normalized


def fit_stimulus_ridge(brain_normalized: np.ndarray,
                       brain_labels: Sequence[Any],
                       unique_labels: Sequence[Any],
                       alpha: float = 100,
                       dtype: Any = np.float32) -> Ridge:
    '''Fit ``Ridge(alpha)`` from normalized brain data onto the stimulus basis.

    The target is the one-hot assignment matrix ``R``, so the fitted model maps
    a brain pattern to ``M`` coefficients, one per unique training stimulus.
    It does not depend on the feature layer.
    '''
    assignment = one_hot_assignment(brain_labels, unique_labels)
    model = Ridge(alpha=alpha)
    model.fit(brain_normalized.astype(dtype), assignment.astype(dtype))
    return model


# Feature-side combination ####################################################

def combine_features(coefficients: np.ndarray, features: np.ndarray,
                     chunk_axis: Optional[int] = None) -> np.ndarray:
    '''Compute ``coefficients @ features`` over the stimulus axis.

    ``features`` has the training features of the unique stimuli along axis 0
    and may be multidimensional (e.g. ``(M, channels, height, width)``).  When
    ``chunk_axis`` is given and the features are at least 3-D, the product is
    evaluated one index at a time along that axis and written straight into the
    output, so no intermediate copy of the whole product is created.  The
    contraction runs over axis 0 only, so every output element is
    mathematically independent of the blocking; it is not bit-identical, since
    BLAS may accumulate a width-1 slice in a different order than the full
    array (a float32 rounding difference, ~1e-7 relative).
    '''
    coefficients = np.asarray(coefficients)
    features = np.asarray(features)

    if coefficients.ndim != 2:
        raise ValueError('coefficients must be 2-D, got %d dimensions'
                         % coefficients.ndim)
    if features.ndim < 2:
        raise ValueError('features must be at least 2-D, got %d dimensions'
                         % features.ndim)
    if coefficients.shape[1] != features.shape[0]:
        raise ValueError(
            'coefficients has %d stimulus columns but features has %d rows'
            % (coefficients.shape[1], features.shape[0]))

    if chunk_axis is None or features.ndim < CHUNK_NDIM + 1:
        return np.tensordot(coefficients, features, axes=([1], [0]))

    out_shape = (coefficients.shape[0],) + features.shape[1:]
    out = np.empty(out_shape, dtype=np.result_type(coefficients, features))
    selector: List[Any] = [slice(None)] * out.ndim
    for i_chunk in range(features.shape[chunk_axis]):
        chunk = np.take(features, [i_chunk], axis=chunk_axis)
        selector[chunk_axis] = slice(i_chunk, i_chunk + 1)
        out[tuple(selector)] = np.tensordot(coefficients, chunk,
                                            axes=([1], [0]))
    return out


# Feature selection ###########################################################

def resolve_feature_index_path(features_path: str,
                               feature_index_file: str) -> str:
    '''Where the feature index of ``features_path`` lives.

    The single resolution rule for the whole pipeline, so that the loader and
    the copy of the index cannot disagree: the index belongs to the feature
    store and is resolved inside ``features_path`` unless it is absolute.
    '''
    if os.path.isabs(feature_index_file):
        return feature_index_file
    return os.path.join(features_path, feature_index_file)


# Feature statistics ##########################################################

def feature_statistics(features: np.ndarray):
    '''``(mean, std)`` of the training features, shaped ``(1, *shape)``.

    The same definition the direct implementation used: per-unit statistics over
    the **unique** stimuli, with ``ddof=1``.
    '''
    mean = np.mean(features, axis=0)[np.newaxis, :]
    std = np.std(features, axis=0, ddof=1)[np.newaxis, :]
    return mean, std


def save_array_atomic(path: str, array: np.ndarray, key: str,
                      dtype: Any = np.float32) -> None:
    '''Write a ``.mat`` array so that concurrent writers cannot tear it.

    The feature statistics are materialized during prediction, and parallel
    prediction workers against one decoder are a supported mode, so two
    processes can reach the same target path.  Writing to a unique temporary
    name in the same directory and then ``os.replace``-ing it is atomic within a
    filesystem: a reader sees either the old file or the complete new one, never
    a partial write, and a failed write leaves any existing file untouched.
    '''
    directory = os.path.dirname(os.path.abspath(path))
    makedir_ifnot(directory)
    # The name must keep the '.mat' suffix: hdf5storage appends one otherwise,
    # and the file would not be where os.replace looks for it.
    temporary = os.path.join(
        directory,
        '.tmp-%d-%s-%s' % (os.getpid(), uuid.uuid4().hex[:8],
                           os.path.basename(path)))
    try:
        save_array(temporary, array, key=key, dtype=dtype, sparse=False)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


# Stimulus correspondence #####################################################

def labels_digest(labels: Sequence[Any]) -> str:
    '''Digest of the **ordered** training labels.

    The stored coefficients are indexed by training stimulus, so the label order
    is what makes ``coeff @ F`` meaningful.  The labels are already part of the
    artifact, so pinning them costs nothing and does not read any feature file.
    A reordered or edited ``train_labels.json`` no longer matches.
    '''
    digest = hashlib.sha256()
    digest.update(b'featdec-labels-v1|')
    for label in labels:
        digest.update(str(label).encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


def verify_train_labels(metadata: Dict[str, Any], train_labels: Sequence[Any],
                        model_dir: str) -> None:
    '''Check the stored labels are the ones the coefficients were fitted with.'''
    spec = metadata.get('labels_digest_spec')
    if spec != LABELS_DIGEST_SPEC:
        raise ValueError(
            '%s records its label digest as %r, but this code computes %r.'
            % (model_dir, spec, LABELS_DIGEST_SPEC))

    actual = labels_digest(train_labels)
    if actual != metadata['labels_digest']:
        raise ValueError(
            'The training labels of %s do not match the order its coefficients '
            'were fitted with (digest %s, expected %s). Each coefficient column '
            'belongs to one training stimulus, so a reordered or edited %s '
            'would silently combine the wrong features.'
            % (model_dir, actual[:16], metadata['labels_digest'][:16],
               LABELS_FILE))


def check_column_correspondence(n_columns: int, train_labels: Sequence[Any],
                                features: np.ndarray,
                                model_dir: str) -> None:
    '''Assert the decoder's only requirement, at the point it is relied on.

    Coefficient column ``i`` belongs to training stimulus ``i``, so it has to
    meet the feature row of that stimulus.  The label order is checked by
    ``verify_train_labels`` and the row assembly by
    ``TrainingFeatureLoader.get``; this catches the remaining way the three can
    disagree -- a model whose width is not the number of stored labels -- with a
    readable message instead of a numpy shape error inside the contraction.
    '''
    n_labels = len(train_labels)
    n_rows = int(np.asarray(features).shape[0])
    if n_columns == n_labels == n_rows:
        return
    raise ValueError(
        'Coefficient column i must be combined with the feature row of '
        'training stimulus i, but %s predicts %d coefficient(s) while %s lists '
        '%d training stimulus/stimuli and %d feature row(s) were loaded.'
        % (model_dir, n_columns, LABELS_FILE, n_labels, n_rows))


# Training feature access #####################################################

class TrainingFeatureLoader(object):
    '''Load unique-stimulus training features, one layer at a time.

    Prediction needs the training features ``F`` of the layer being decoded.
    Holding one layer costs the same memory the training step already needs, but
    holding several would not scale, so this cache keeps **exactly one** layer
    resident: requesting a different layer releases the previous array first.
    '''

    def __init__(self, features_paths: Sequence[str],
                 feature_index_file: Optional[str] = None):
        self._paths = list(features_paths)
        self._feature_index_file = feature_index_file
        self._stores: Optional[List[Features]] = None
        self._cache_key: Optional[Tuple[str, Tuple[Any, ...]]] = None
        self._cache: Optional[np.ndarray] = None
        self.load_count = 0  # number of times features were read from disk

    @property
    def cached_layers(self) -> List[str]:
        '''Layers currently held in memory (never more than one).'''
        return [] if self._cache_key is None else [self._cache_key[0]]

    def features(self) -> List[Features]:
        if self._stores is None:
            if self._feature_index_file is None:
                self._stores = [Features(path) for path in self._paths]
            else:
                self._stores = [
                    Features(path, feature_index=resolve_feature_index_path(
                        path, self._feature_index_file))
                    for path in self._paths]
        return self._stores

    def get(self, layer: str, labels: Sequence[Any]) -> np.ndarray:
        '''Features of ``labels`` for ``layer``, shape ``(len(labels), ...)``.

        Row ``i`` is the features of ``labels[i]`` -- which is what makes the
        coefficients meaningful, so it is enforced per label rather than checked
        in aggregate.  ``bdpy.dataform.utils.get_multi_features`` cannot be used
        for this: it appends a row for *every* store holding a label and skips
        labels no store holds, so one duplicated and one missing label cancel
        out in the total row count while every row after them is shifted against
        its coefficient column.
        '''
        key = (layer, tuple(labels))
        if self._cache_key == key:
            assert self._cache is not None
            return self._cache

        # Release the previous layer before allocating the next one.
        self.release()

        rows = []
        for label in labels:
            sources = [store for store in self.features()
                       if label in store.labels]
            if len(sources) != 1:
                raise ValueError(
                    'Training stimulus %r must come from exactly one feature '
                    'store, but %d of %d store(s) provide it: %s'
                    % (label, len(sources), len(self._paths),
                       ', '.join(self._paths)))
            row = sources[0].get(layer=layer, label=label)
            if row.shape[0] != 1:
                raise ValueError(
                    'Training stimulus %r contributes %d feature rows for '
                    'layer %r; one row per stimulus is required.'
                    % (label, row.shape[0], layer))
            rows.append(row)

        features = np.vstack(rows)
        self._cache_key = key
        self._cache = features
        self.load_count += 1
        return features

    def release(self) -> None:
        '''Drop the cached layer.'''
        self._cache = None
        self._cache_key = None


# Serialization ###############################################################

def brain_model_dir(decoder_path: str, subject: str, roi: str) -> str:
    """Where the brain-side model of one (subject, ROI) lives."""
    return os.path.join(decoder_path, subject, roi, 'model')


def layer_model_dir(decoder_path: str, layer: str, subject: str,
                    roi: str) -> str:
    """The per-layer directory: a legacy decoder, or the statistics sidecars.

    ``evaluation.py`` reads ``y_mean``/``y_norm`` from exactly this path.
    """
    return os.path.join(decoder_path, layer, subject, roi, 'model')


def is_factorized_model_dir(model_dir: str) -> bool:
    '''True if ``model_dir`` holds a factorized decoder.

    The presence of ``factorized.yaml`` is the only discriminator, and it is
    checked on the **brain-side** directory. A factorized decoder's per-layer
    directories contain nothing but the statistics sidecars, which from the
    outside look exactly like a legacy decoder directory, so deciding by the
    per-layer path would misread a decoder as legacy as soon as prediction has
    run against it once.
    '''
    return os.path.isfile(os.path.join(model_dir, META_FILE))


def save_factorized_model(model_dir: str, model: Ridge,
                          train_labels: Sequence[Any],
                          metadata: Optional[Dict[str, Any]] = None) -> List[str]:
    '''Write the factorized decoder into ``model_dir``.

    The estimator itself is stored in bdpy's ``ModelTraining`` pickle layout
    (``{'model': ..., 'y_shape': ...}``) so that ``bdpy.ml.ModelTest`` reads it
    without modification.  ``train_labels`` records the stimulus order, i.e. the
    row order of ``F`` the coefficients refer to.  ``factorized.yaml`` marks the
    format and is what distinguishes a factorized decoder from a legacy one.
    '''
    makedir_ifnot(model_dir)
    train_labels = [str(label) for label in train_labels]

    model_path = os.path.join(model_dir, MODEL_FILE)
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'y_shape': (len(train_labels),)}, f,
                    protocol=PICKLE_PROTOCOL)

    labels_path = os.path.join(model_dir, LABELS_FILE)
    with open(labels_path, 'w') as f:
        json.dump(train_labels, f, indent=1)

    meta: Dict[str, Any] = {
        'format': FORMAT_NAME,
        'version': FORMAT_VERSION,
        'model_file': MODEL_FILE,
        'train_labels_file': LABELS_FILE,
        'n_train_stimuli': len(train_labels),
    }
    meta.update(metadata or {})
    meta_path = os.path.join(model_dir, META_FILE)
    with open(meta_path, 'w') as f:
        f.write(yaml.dump(meta, default_flow_style=False, sort_keys=True))

    return [model_path, labels_path, meta_path]


def load_factorized_metadata(model_dir: str) -> Dict[str, Any]:
    '''Read ``factorized.yaml``.'''
    with open(os.path.join(model_dir, META_FILE), 'r') as f:
        meta = yaml.safe_load(f)
    if not isinstance(meta, dict) or meta.get('format') != FORMAT_NAME:
        raise ValueError('%s is not a %s decoder' % (model_dir, FORMAT_NAME))
    if meta.get('version') != FORMAT_VERSION:
        raise ValueError(
            'Unsupported %s version %r in %s (this code writes version %d)'
            % (FORMAT_NAME, meta.get('version'), model_dir, FORMAT_VERSION))
    return meta


def load_train_labels(model_dir: str) -> List[str]:
    '''Read the stimulus order the stored coefficients refer to.'''
    meta = load_factorized_metadata(model_dir)
    labels_file = meta.get('train_labels_file', LABELS_FILE)
    with open(os.path.join(model_dir, labels_file), 'r') as f:
        labels = json.load(f)
    if len(labels) != meta['n_train_stimuli']:
        raise ValueError(
            '%s lists %d labels but the metadata says %d'
            % (labels_file, len(labels), meta['n_train_stimuli']))
    return [str(label) for label in labels]


def model_file_path(model_dir: str) -> str:
    '''Path of the pickled brain-side Ridge model.'''
    meta = load_factorized_metadata(model_dir)
    return os.path.join(model_dir, meta.get('model_file', MODEL_FILE))
