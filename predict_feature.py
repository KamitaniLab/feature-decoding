'''DNN Feature decoding - feature prediction script.

Decoders trained by ``train_decoder_sklearn_ridge.py`` are factorized (see
``ridge_factorization``), so prediction runs in two steps: predict one
coefficient per training stimulus, then combine those coefficients with the
training features of the layer being decoded.  Decoders saved by the earlier,
direct implementation are detected automatically and predicted as before.
'''


from typing import Optional, Sequence

from itertools import product
import os
import shutil
from time import time
import warnings

import bdpy
from bdpy.dataform import load_array, save_array
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTest
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.util import makedir_ifnot
import numpy as np

from ridge_factorization import (
    FEATURE_INDEX_FILE,
    TrainingFeatureLoader,
    brain_model_dir,
    check_column_correspondence,
    combine_features,
    feature_statistics,
    is_factorized_model_dir,
    load_factorized_metadata,
    layer_model_dir,
    load_train_labels,
    model_file_path,
    resolve_feature_index_path,
    save_array_atomic,
    verify_train_labels,
)


# Prediction back ends #######################################################

def _training_features(model_dir: str, layer: str,
                       feature_loader: TrainingFeatureLoader):
    """Load the training features the decoder's coefficients refer to.

    Returns ``(train_labels, features)``, one feature row per training stimulus
    in the order the coefficient columns were fitted in.
    """
    metadata = load_factorized_metadata(model_dir)
    train_labels = load_train_labels(model_dir)
    verify_train_labels(metadata, train_labels, model_dir)
    return train_labels, feature_loader.get(layer, train_labels)


class FeatureStatistics(object):
    """``y_mean``/``y_norm`` of the training features, cached per label set.

    They depend on ``(layer, ordered training labels)`` only -- never on the
    ROI -- and at DeepRecon scale each computation is two passes over a 15 GB
    tensor, so they are computed once and reused for every decoder directory.
    """

    def __init__(self):
        self._cache = {}
        self.computations = 0  # observable in tests

    def get(self, layer: str, train_labels: Sequence[str],
            features: np.ndarray):
        key = (layer, tuple(train_labels))
        if key not in self._cache:
            self._cache[key] = feature_statistics(features)
            self.computations += 1
        return self._cache[key]


def _materialize_feature_statistics(decoder_path: str, layer: str,
                                    subject: str, roi: str, model_dir: str,
                                    feature_loader: TrainingFeatureLoader,
                                    statistics: FeatureStatistics) -> None:
    """Fill in the ``y_mean``/``y_norm`` that ``evaluation.py`` reads.

    Compatibility sidecars, not parameters of the model: prediction never reads
    them, and training cannot produce them because it never opens a feature
    file, so the step that has the features writes them.  Only when missing --
    one decoder directory belongs to one training feature configuration -- and
    atomically, since parallel prediction workers can reach the same path.
    """
    directory = layer_model_dir(decoder_path, layer, subject, roi)
    targets = {key: os.path.join(directory, '%s.mat' % key)
               for key in ('y_mean', 'y_norm')}
    if all(os.path.exists(path) for path in targets.values()):
        return

    train_labels, features = _training_features(model_dir, layer,
                                                feature_loader)
    values = dict(zip(('y_mean', 'y_norm'),
                      statistics.get(layer, train_labels, features)))

    for key, path in sorted(targets.items()):
        if os.path.exists(path):
            continue
        try:
            save_array_atomic(path, values[key], key=key, dtype=np.float32)
            print('Saved %s' % path)
        except Exception:
            warnings.warn('Failed to save %s. The decoder directory may be '
                          'read-only; evaluation.py will not find the training '
                          'feature statistics.' % path)


def _predict_factorized(model_dir: str, brain: np.ndarray,
                        layer: str, chunk_axis: Optional[int],
                        feature_loader: TrainingFeatureLoader) -> np.ndarray:
    '''Predict features with a factorized decoder.

    ``brain`` must already be normalized with the decoder's ``x_mean`` /
    ``x_norm``.  The feature normalization parameters are not needed: they
    cancel algebraically (``ridge_factorization``), so the prediction is the
    linear combination of the raw training features.
    '''
    train_labels, features = _training_features(model_dir, layer,
                                                feature_loader)

    test = ModelTest(None, brain)
    test.model_format = 'pickle'
    test.model_path = model_file_path(model_dir)
    test.dtype = np.float32
    test.chunk_axis = None  # the target is the stimulus basis; never chunked

    coefficients = test.run()
    check_column_correspondence(np.asarray(coefficients).shape[1],
                                train_labels, features, model_dir)

    # np.asarray rather than .astype: the training features can be very large
    # and are already float32, so this must not copy them.
    return combine_features(np.asarray(coefficients, dtype=np.float32),
                            np.asarray(features, dtype=np.float32),
                            chunk_axis=chunk_axis)


def _predict_legacy(model_dir: str, brain: np.ndarray,
                    chunk_axis: Optional[int]) -> np.ndarray:
    '''Predict features with a decoder saved by the direct implementation.

    One Ridge model per chunk of the feature dimensions, and the prediction is
    in normalized feature space, so it has to be un-normalized here.
    '''
    feat_mean = load_array(os.path.join(model_dir, 'y_mean.mat'), key='y_mean')  # shape = (1, shape_features)
    feat_norm = load_array(os.path.join(model_dir, 'y_norm.mat'), key='y_norm')  # shape = (1, shape_features)

    test = ModelTest(None, brain)
    test.model_format = 'pickle'
    test.model_path = model_dir
    test.dtype = np.float32
    test.chunk_axis = chunk_axis

    feat_pred = test.run()

    return feat_pred * feat_norm + feat_mean


# Main #######################################################################

def featdec_predict(
        fmri_data,
        decoder_path,
        output_dir='./decoded_features',
        rois=None,
        label_key=None,
        layers=None,
        feature_index_file=None,
        excluded_labels=[],
        average_sample=True,
        chunk_axis=1,
        training_features_paths=None,
        analysis_name="feature_prediction"
):
    '''Feature prediction.

    Input:

    - fmri_data
    - feature_decoder_dir
    - training_features_paths: directories holding the features of the training
      stimuli.  Required for decoders in the factorized format.

    Output:

    - output_dir

    Parameters:

    TBA
    '''
    layers = layers[::-1]  # Start training from deep layers

    # Print info -------------------------------------------------------------
    print('Subjects:        %s' % list(fmri_data.keys()))
    print('ROIs:            %s' % list(rois.keys()))
    print('Decoders:        %s' % decoder_path)
    print('Layers:          %s' % layers)
    print('')

    # Load data --------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_brain = {sbj: bdpy.BData(dat_file[0])
                  for sbj, dat_file in fmri_data.items()}

    # Training features of the factorized decoders.  Only the layer currently
    # being decoded is held in memory (see TrainingFeatureLoader).
    feature_loader = TrainingFeatureLoader(
        training_features_paths or [], feature_index_file=feature_index_file)
    statistics = FeatureStatistics()

    # Initialize directories -------------------------------------------
    makedir_ifnot(output_dir)
    makedir_ifnot('tmp')

    # Save feature index -----------------------------------------------------
    if feature_index_file is not None:
        # The path as given wins; only when it does not resolve is the feature
        # store consulted, which is where a factorized decoder's index lives
        # (`TrainingFeatureLoader` hands the same resolved path to `Features`).
        feature_index_source = feature_index_file
        if not os.path.exists(feature_index_source) and training_features_paths:
            feature_index_source = resolve_feature_index_path(
                training_features_paths[0], feature_index_file)
        feature_index_save_file = os.path.join(output_dir, FEATURE_INDEX_FILE)
        shutil.copy(feature_index_source, feature_index_save_file)
        print('Saved %s' % feature_index_save_file)

    # Analysis loop ----------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    # `layer` is the outer loop so that the training features of one layer are
    # read once and reused for every subject and ROI.
    current_layer = None
    for layer, sbj, roi in product(layers, fmri_data, rois):
        if layer != current_layer:
            # Release the previous layer's training features before loading the
            # next ones: at most one layer is ever resident.
            feature_loader.release()
            current_layer = layer

        print('--------------------')
        print('Feature:    %s' % layer)
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)

        # Distributed computation setup
        # -----------------------------
        analysis_id = analysis_name + '-' + sbj + '-' + roi + '-' + layer
        results_dir_prediction = os.path.join(output_dir, layer, sbj, roi)

        # The brain-side directory is the only discriminator: once prediction
        # has run once, a factorized decoder's per-layer directory looks like a
        # legacy decoder directory from the outside.
        shared_dir = brain_model_dir(decoder_path, sbj, roi)
        factorized = is_factorized_model_dir(shared_dir)
        model_dir = (shared_dir if factorized
                     else layer_model_dir(decoder_path, layer, sbj, roi))

        # Before the "already done" skip below: a run whose features were
        # already decoded would otherwise skip forever and the statistics would
        # never appear. Only for factorized decoders -- a legacy decoder wrote
        # its own at training time -- and only when the features are available.
        if factorized and training_features_paths:
            _materialize_feature_statistics(
                decoder_path, layer, sbj, roi, model_dir, feature_loader,
                statistics)

        if os.path.exists(results_dir_prediction):
            print('%s is already done. Skipped.' % analysis_id)
            continue

        # Before `makedir_ifnot`: a run that cannot proceed must not leave an
        # output directory behind, or the next, correct run would skip it.
        if factorized and not training_features_paths:
            raise ValueError(
                '%s is a factorized decoder, which needs the features of the '
                'training stimuli. Pass training_features_paths (the '
                'decoder.features.paths of the config used for training).'
                % model_dir)

        makedir_ifnot(results_dir_prediction)

        distcomp_db = os.path.join('./tmp', analysis_name + '.db')
        distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)
        if not distcomp.lock(analysis_id):
            print('%s is already running. Skipped.' % analysis_id)
            continue

        # Preparing data
        # --------------
        print('Preparing data')

        start_time = time()

        # Brain data
        brain = data_brain[sbj].select(rois[roi])
        # TODO: Dirty solution. FIXME!
        try:
            brain_labels = data_brain[sbj].get_label(label_key)
        except ValueError:
            print(f'{label_key} not found in vmap. Select numerical values of {label_key} as labels.')
            brain_labels = list(data_brain[sbj].select(label_key).flatten())

        # Averaging brain data
        if average_sample:
            brain_labels_unique = np.unique(brain_labels)
            brain_labels_unique = [lb for lb in brain_labels_unique if lb not in excluded_labels]
            brain = np.vstack([np.mean(brain[(np.array(brain_labels) == lb).flatten(), :], axis=0) for lb in brain_labels_unique])
        else:
            # Sample No. + Label
            brain_labels_unique = ['sample{:06}-{}'.format(i + 1, lb) for i, lb in enumerate(brain_labels)]

        print('Elapsed time (data preparation): %f' % (time() - start_time))

        # Preprocessing
        # -------------
        brain_mean = load_array(os.path.join(model_dir, 'x_mean.mat'), key='x_mean')  # shape = (1, n_voxels)
        brain_norm = load_array(os.path.join(model_dir, 'x_norm.mat'), key='x_norm')  # shape = (1, n_voxels)

        brain = (brain - brain_mean) / brain_norm

        # Prediction
        # ----------
        print('Prediction')

        start_time = time()

        if factorized:
            feat_pred = _predict_factorized(
                model_dir, brain, layer, chunk_axis, feature_loader)
        else:
            print('Legacy decoder format detected in %s' % model_dir)
            feat_pred = _predict_legacy(model_dir, brain, chunk_axis)

        print('Total elapsed time (prediction): %f' % (time() - start_time))

        # Save results
        # ------------
        print('Saving results')

        start_time = time()

        # Predicted features
        for i, label in enumerate(brain_labels_unique):
            # Predicted features
            _feat = np.array([feat_pred[i,]])  # To make feat shape 1 x M x N x ...

            # Save file name
            save_file = os.path.join(results_dir_prediction, '%s.mat' % label)

            # Save
            save_array(save_file, _feat, key='feat', dtype=np.float32, sparse=False)

        print('Saved %s' % results_dir_prediction)

        print('Elapsed time (saving results): %f' % (time() - start_time))

        distcomp.unlock(analysis_id)

    feature_loader.release()

    print('%s finished.' % analysis_name)

    return output_dir


# Entry point ################################################################

if __name__ == '__main__':

    cfg = init_hydra_cfg()

    analysis_name = cfg["_run_"]["name"] + '-' + cfg["_run_"]["config_name"]

    decoder_path = cfg["decoded_feature"]["decoder"]["path"]

    test_fmri_data = {
        subject["name"]: subject["paths"]
        for subject in cfg["decoded_feature"]["fmri"]["subjects"]
    }
    rois = {
        roi["name"]: roi["select"]
        for roi in cfg["decoded_feature"]["fmri"]["rois"]
    }
    label_key = cfg["decoded_feature"]["fmri"]["label_key"]

    layers = cfg["decoded_feature"]["features"]["layers"]
    feature_index_file = cfg.decoder.features.get("index_file", None)

    decoded_feature_dir = cfg["decoded_feature"]["path"]

    average_sample = cfg["decoded_feature"]["parameters"]["average_sample"]
    excluded_labels = cfg.decoded_feature.fmri.get("exclude_labels", [])

    # Features of the training stimuli. The factorized decoder references them
    # instead of storing a copy, so prediction needs them too.
    training_features_paths = cfg.decoded_feature.decoder.get(
        "training_features_paths", None)
    if training_features_paths is None:
        training_features_paths = cfg["decoder"]["features"]["paths"]

    featdec_predict(
        test_fmri_data,
        decoder_path,
        output_dir=decoded_feature_dir,
        rois=rois,
        label_key=label_key,
        layers=layers,
        feature_index_file=feature_index_file,
        excluded_labels=excluded_labels,
        average_sample=average_sample,
        chunk_axis=cfg["decoder"]["parameters"]["chunk_axis"],
        training_features_paths=training_features_paths,
        analysis_name=analysis_name
    )
