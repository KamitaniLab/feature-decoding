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

import predict_feature
import train_decoder_sklearn_ridge

NORM_KEYS = ('x_mean', 'x_norm', 'y_mean', 'y_norm')


def run_training(dataset, output_dir: str, alpha: float = 100,
                 chunk_axis: Optional[int] = 1,
                 layers: Optional[Sequence[str]] = None,
                 subjects: Optional[Sequence[str]] = None,
                 rois: Optional[Dict[str, str]] = None,
                 analysis_name: str = 'test_training') -> str:
    """Train decoders with ``train_decoder_sklearn_ridge.py``."""
    fmri = {sbj: paths for sbj, paths in dataset.train_fmri.items()
            if subjects is None or sbj in subjects}
    return train_decoder_sklearn_ridge.featdec_sklearn_ridge_train(
        fmri,
        [dataset.train_features_dir],
        output_dir=output_dir,
        rois=dict(dataset.rois) if rois is None else dict(rois),
        label_key=dataset.label_key,
        layers=list(dataset.layers if layers is None else layers),
        alpha=alpha,
        chunk_axis=chunk_axis,
        analysis_name=analysis_name,
    )


def run_prediction(dataset, decoder_dir: str, output_dir: str,
                   chunk_axis: Optional[int] = 1,
                   layers: Optional[Sequence[str]] = None,
                   subjects: Optional[Sequence[str]] = None,
                   rois: Optional[Dict[str, str]] = None,
                   average_sample: bool = True,
                   excluded_labels: Sequence[str] = (),
                   analysis_name: str = 'test_prediction') -> str:
    """Predict features with ``predict_feature.py``.

    ``training_features_paths`` is passed only when the script accepts it, so
    this helper works both before and after the decoder is factorized.
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
        analysis_name=analysis_name,
    )
    if 'training_features_paths' in inspect.signature(
            predict_feature.featdec_predict).parameters:
        kwargs['training_features_paths'] = [dataset.train_features_dir]
    return predict_feature.featdec_predict(fmri, decoder_dir, **kwargs)


def model_dir(decoder_dir: str, layer: str, subject: str, roi: str) -> str:
    return os.path.join(decoder_dir, layer, subject, roi, 'model')


def read_norm_params(decoder_dir: str, layer: str, subject: str,
                     roi: str) -> Dict[str, np.ndarray]:
    directory = model_dir(decoder_dir, layer, subject, roi)
    return {key: load_array(os.path.join(directory, key + '.mat'), key=key)
            for key in NORM_KEYS}


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
