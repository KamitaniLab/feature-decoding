"""Run the real training / prediction scripts on a synthetic dataset.

Shared by the tests and by ``tests/generate_golden.py`` so that the golden
fixtures and the tests exercise exactly the same code path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import inspect
import os

import numpy as np
from bdpy.dataform import load_array

import evaluation
import predict_feature
import ridge_factorization
import train_decoder_sklearn_ridge

BRAIN_NORM_KEYS = ('x_mean', 'x_norm')
FEATURE_NORM_KEYS = ('y_mean', 'y_norm')


def run_training(dataset, output_dir: str, alpha: float = 100,
                 subjects: Optional[Sequence[str]] = None,
                 rois: Optional[Dict[str, str]] = None,
                 analysis_name: str = 'test_training') -> str:
    """Train decoders with ``train_decoder_sklearn_ridge.py``.

    No feature-side argument at all -- not even the layers: the factorized fit
    is on the stimulus basis and is shared by every layer.
    """
    fmri = {sbj: paths for sbj, paths in dataset.train_fmri.items()
            if subjects is None or sbj in subjects}
    return train_decoder_sklearn_ridge.featdec_sklearn_ridge_train(
        fmri,
        output_dir=output_dir,
        rois=dict(dataset.rois) if rois is None else dict(rois),
        label_key=dataset.label_key,
        alpha=alpha,
        analysis_name=analysis_name,
    )


def run_prediction(dataset, decoder_dir: str, output_dir: str,
                   chunk_axis: Optional[int] = 1,
                   layers: Optional[Sequence[str]] = None,
                   subjects: Optional[Sequence[str]] = None,
                   rois: Optional[Dict[str, str]] = None,
                   average_sample: bool = True,
                   excluded_labels: Sequence[str] = (),
                   features_paths: Optional[Sequence[str]] = None,
                   feature_index_file: Optional[str] = None,
                   analysis_name: str = 'test_prediction') -> str:
    """Predict features with ``predict_feature.py``.

    ``training_features_paths`` is passed only when the script accepts it, so
    this helper works both before and after the decoder is factorized.  Pass
    ``features_paths=[]`` to run without them.
    """
    fmri = {sbj: paths for sbj, paths in dataset.test_fmri.items()
            if subjects is None or sbj in subjects}
    kwargs: Dict[str, Any] = dict(
        output_dir=output_dir,
        rois=dict(dataset.rois) if rois is None else dict(rois),
        label_key=dataset.label_key,
        layers=list(dataset.layers if layers is None else layers),
        excluded_labels=list(excluded_labels),
        average_sample=average_sample,
        chunk_axis=chunk_axis,
        feature_index_file=feature_index_file,
        analysis_name=analysis_name,
    )
    if 'training_features_paths' in inspect.signature(
            predict_feature.featdec_predict).parameters:
        # An explicit empty list means "run without the training features",
        # which is a supported case for an already-finished prediction.
        if features_paths is None:
            features_paths = [dataset.train_features_dir]
        kwargs['training_features_paths'] = list(features_paths)
    return predict_feature.featdec_predict(fmri, decoder_dir, **kwargs)


def brain_model_dir(decoder_dir: str, subject: str, roi: str) -> str:
    """The shared brain-side model of a factorized decoder."""
    return ridge_factorization.brain_model_dir(decoder_dir, subject, roi)


def layer_model_dir(decoder_dir: str, layer: str, subject: str,
                    roi: str) -> str:
    """A legacy decoder directory, or a factorized decoder's sidecars."""
    return ridge_factorization.layer_model_dir(decoder_dir, layer, subject, roi)


def read_norm_params(decoder_dir: str, subject: str, roi: str,
                     keys: Sequence[str] = BRAIN_NORM_KEYS
                     ) -> Dict[str, np.ndarray]:
    """Load the brain-side normalization parameters of a factorized decoder."""
    directory = brain_model_dir(decoder_dir, subject, roi)
    return {key: load_array(os.path.join(directory, key + '.mat'), key=key)
            for key in keys}


def read_feature_statistics(decoder_dir: str, layer: str, subject: str,
                            roi: str) -> Dict[str, np.ndarray]:
    """Load the per-layer statistics sidecars ``evaluation.py`` reads."""
    directory = layer_model_dir(decoder_dir, layer, subject, roi)
    return {key: load_array(os.path.join(directory, key + '.mat'), key=key)
            for key in FEATURE_NORM_KEYS}


def read_legacy_norm_params(decoder_dir: str, layer: str, subject: str,
                            roi: str) -> Dict[str, np.ndarray]:
    """All four parameters of a decoder in the direct (per-layer) format."""
    directory = layer_model_dir(decoder_dir, layer, subject, roi)
    return {key: load_array(os.path.join(directory, key + '.mat'), key=key)
            for key in BRAIN_NORM_KEYS + FEATURE_NORM_KEYS}


def run_evaluation(dataset, decoder_dir: str, decoded_dir: str,
                   true_features_dir: str,
                   layers: Optional[Sequence[str]] = None,
                   subjects: Optional[Sequence[str]] = None,
                   rois: Optional[Dict[str, str]] = None,
                   output_file: Optional[str] = None,
                   feature_index_file: Optional[str] = None,
                   average_sample: bool = True) -> str:
    """Evaluate decoded features with ``evaluation.py``."""
    return evaluation.featdec_eval(
        decoded_dir,
        true_features_dir,
        output_file=output_file or os.path.join(decoded_dir, 'evaluation.db'),
        subjects=list(subjects or dataset.test_fmri),
        rois=list(rois or dataset.rois),
        layers=list(dataset.layers if layers is None else layers),
        feature_index_file=feature_index_file,
        feature_decoder_path=decoder_dir,
        average_sample=average_sample,
    )


def read_decoded_features(output_dir: str, layer: str, subject: str, roi: str,
                          labels: Sequence[str]) -> np.ndarray:
    """Stack the saved per-label ``.mat`` files into one array."""
    directory = os.path.join(output_dir, layer, subject, roi)
    return np.vstack([
        load_array(os.path.join(directory, '%s.mat' % label), key='feat')
        for label in labels])


def decoded_feature_labels(output_dir: str, layer: str, subject: str,
                           roi: str) -> List[str]:
    directory = os.path.join(output_dir, layer, subject, roi)
    return sorted(os.path.splitext(f)[0] for f in os.listdir(directory)
                  if f.endswith('.mat'))
