'''DNN Feature decoding - decoders training script.

Trains a scikit-learn ``Ridge`` decoder in the factorized form described in
``ridge_factorization``: the model maps brain activity onto the training
stimulus basis, so the fit needs the trial labels only.  This script reads no
feature data and takes no feature-side parameter.
'''


from typing import Any, Dict, List, Optional, Sequence

from itertools import product
import os
from time import time
import warnings

import bdpy
from bdpy.bdata.utils import select_data_multi_bdatas, get_labels_multi_bdatas
from bdpy.dataform import save_array
from bdpy.distcomp import DistComp
from bdpy.pipeline.config import init_hydra_cfg
from bdpy.util import makedir_ifnot
import numpy as np
import yaml

from ridge_factorization import (
    LABELS_DIGEST_SPEC,
    brain_model_dir,
    fit_stimulus_ridge,
    labels_digest,
    normalize_brain_for_training,
    save_factorized_model,
)


DTYPE = np.float32


# Helpers ####################################################################

def _load_info(info_file: str) -> Dict[str, Any]:
    '''Read a model ``info.yaml``, retrying while it reads back as empty.'''
    with open(info_file, 'r') as f:
        info = yaml.safe_load(f)
    while info is None:
        warnings.warn('Failed to load info from %s. Retrying...'
                      % info_file)
        with open(info_file, 'r') as f:
            info = yaml.safe_load(f)
    return info


def _is_done(model_dir: str) -> bool:
    '''True if ``model_dir`` holds a completed decoder.'''
    info_file = os.path.join(model_dir, 'info.yaml')
    if not os.path.exists(info_file):
        return False
    info = _load_info(info_file)
    status = info.get('_status', {}) if isinstance(info, dict) else {}
    return status.get('computation_status') == 'done'


def _mark_done(model_dir: str, computation_id: str) -> None:
    '''Record the completion status in ``info.yaml``.

    ``bdpy.ml.ModelTraining`` used to write this; the factorized training does
    not go through it, so the same marker is written here instead.
    '''
    info_file = os.path.join(model_dir, 'info.yaml')
    info = _load_info(info_file) if os.path.exists(info_file) else {}
    if not isinstance(info, dict):
        info = {}
    info.setdefault('_status', {})
    info['_status'].update({
        'computation_id': computation_id,
        'computation_status': 'done',
    })
    with open(info_file, 'w') as f:
        f.write(yaml.dump(info, default_flow_style=False))


def _save_norm_params(model_dir: str,
                      norm_param: Dict[str, np.ndarray]) -> None:
    '''Save the brain normalization parameters ``x_mean``/``x_norm``.

    The feature statistics are written by ``predict_feature.py``: this script
    never opens a feature file.
    '''
    for key in sorted(norm_param):
        save_file = os.path.join(model_dir, key + '.mat')
        if not os.path.exists(save_file):
            try:
                save_array(save_file, norm_param[key], key=key, dtype=DTYPE,
                           sparse=False)
                print('Saved %s' % save_file)
            except Exception:
                warnings.warn('Failed to save %s. Possibly double running.'
                              % save_file)


def _write_brain_decoder(model_dir: str, model, feat_labels: Sequence[Any],
                         brain_mean: np.ndarray, brain_norm: np.ndarray,
                         alpha: float, subject: str, roi: str,
                         n_trials: int, n_voxels: int) -> None:
    """Write the brain-side decoder of one (subject, ROI)."""
    metadata = {
        'alpha': float(alpha),
        'dtype': np.dtype(DTYPE).name,
        'labels_digest': labels_digest(feat_labels),
        'labels_digest_spec': LABELS_DIGEST_SPEC,
        'n_trials': int(n_trials),
        'n_voxels': int(n_voxels),
        'roi': str(roi),
        'subject': str(subject),
    }

    # Save normalization parameters
    # -----------------------------
    print('Saving normalization parameters.')
    _save_norm_params(model_dir, {'x_mean': brain_mean, 'x_norm': brain_norm})

    # Save the model
    # --------------
    save_factorized_model(model_dir, model, feat_labels, metadata=metadata)
    print('Saved %s' % model_dir)


# Main #######################################################################

def featdec_sklearn_ridge_train(
        fmri_data: Dict[str, List[str]],
        output_dir: str = './feature_decoders',
        rois: Optional[Dict[str, str]] = None,
        label_key: Optional[str] = None,
        alpha: int = 100,
        analysis_name: str = "feature_decoder_training"
):
    '''Feature decoder training.

    Input:

    - fmri_data

    Output:

    - output_dir

    Note:

    One model per (subject, ROI), shared by every feature layer, written to
    `<output_dir>/<subject>/<roi>/model/`.
    '''
    if rois is None:
        rois = {}

    # Print info -------------------------------------------------------------
    print('Subjects:        %s' % list(fmri_data.keys()))
    print('ROIs:            %s' % list(rois.keys()))
    print('')

    # Load data --------------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_brain = {sbj: [bdpy.BData(f) for f in data_files] for sbj, data_files in fmri_data.items()}

    # Initialize directories -------------------------------------------------
    makedir_ifnot(output_dir)
    makedir_ifnot('tmp')

    # Distributed computation setup ------------------------------------------
    distcomp_db = os.path.join('./tmp', analysis_name + '.db')
    distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)

    # Analysis loop ----------------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for sbj, roi in np.random.permutation(list(product(fmri_data.keys(), rois.keys()))):
        print('--------------------')
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)

        analysis_id = analysis_name + '-' + sbj + '-' + roi
        model_dir = brain_model_dir(output_dir, sbj, roi)

        if _is_done(model_dir):
            print('%s is already done and skipped' % analysis_id)
            continue

        makedir_ifnot(model_dir)

        # DistComp.lock() returns True if the computation is not locked and
        # successfully locked.
        if not distcomp.lock(analysis_id):
            print('%s is already running. Skipped.' % analysis_id)
            continue

        # Preparing data
        # --------------
        print('Preparing data')

        start_time = time()

        # Brain data
        brain = select_data_multi_bdatas(data_brain[sbj], rois[roi])
        brain_labels = get_labels_multi_bdatas(data_brain[sbj], label_key)

        # Unique training stimuli, in the order the training features are read.
        feat_labels = np.unique(brain_labels)

        # Use brain data that has a label included in feature data
        brain = np.vstack([_b for _b, bl in zip(brain, brain_labels) if bl in feat_labels])
        brain_labels = [bl for bl in brain_labels if bl in feat_labels]

        print('Elapsed time (data preparation): %f' % (time() - start_time))

        # Calculate normalization parameters
        # ----------------------------------

        # Normalize brain data
        brain_mean = np.mean(brain, axis=0)[np.newaxis, :]  # np.newaxis was added to match Matlab outputs
        brain_norm = np.std(brain, axis=0, ddof=1)[np.newaxis, :]

        # Model training
        # --------------
        print('Model training (brain -> training stimulus basis)')
        start_time = time()

        brain_normalized = normalize_brain_for_training(brain, brain_mean,
                                                        brain_norm)
        model = fit_stimulus_ridge(brain_normalized, brain_labels, feat_labels,
                                   alpha=alpha, dtype=DTYPE)

        print('Elapsed time (model training): %f' % (time() - start_time))

        _write_brain_decoder(
            model_dir, model, feat_labels, brain_mean, brain_norm, alpha,
            sbj, roi, n_trials=brain.shape[0], n_voxels=brain.shape[1])

        _mark_done(model_dir, analysis_id)

        distcomp.unlock(analysis_id)

    print('%s finished.' % analysis_name)

    return output_dir


# Entry point ################################################################

if __name__ == '__main__':

    cfg = init_hydra_cfg()

    analysis_name = cfg["_run_"]["name"] + '-' + cfg["_run_"]["config_name"]

    training_fmri = {
        subject["name"]: subject["paths"]
        for subject in cfg["decoder"]["fmri"]["subjects"]
    }
    rois = {
        roi["name"]: roi["select"]
        for roi in cfg["decoder"]["fmri"]["rois"]
    }
    label_key = cfg["decoder"]["fmri"]["label_key"]

    # Nothing under `decoder.features` is read here; `predict_feature.py`
    # consumes those keys.
    decoder_dir = cfg["decoder"]["path"]

    featdec_sklearn_ridge_train(
        training_fmri,
        output_dir=decoder_dir,
        rois=rois,
        label_key=label_key,
        alpha=cfg["decoder"]["parameters"]["alpha"],
        analysis_name=analysis_name
    )
