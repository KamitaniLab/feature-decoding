"""The factorized pipeline still satisfies the evaluation file contract.

``evaluation.py`` is not changed by the factorization, and it reads the
training-feature statistics from
``<decoder>/<layer>/<subject>/<roi>/model/y_{mean,norm}.mat``.  The factorized
decoder cannot write them at training time -- it never opens a feature file --
so ``predict_feature.py`` fills them in.  This test exists to catch that
contract breaking: nothing else in the suite runs ``evaluation.py``.
"""

from __future__ import annotations

import os

import numpy as np

from tests.helpers import pipeline, synthetic

ALPHA = 100
CHUNK_AXIS = 1
LAYER = 'fc_like'
ONE_SUBJECT = ['sub-01']
ONE_ROI = {'VC': 'ROI_VC = 1'}


def test_train_predict_evaluate_end_to_end(dataset, tmp_path):
    """Train, predict, then run the unchanged ``evaluation.py`` over it."""
    decoder_dir = str(tmp_path / 'decoders')
    decoded_dir = str(tmp_path / 'decoded')

    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA,
                          subjects=ONE_SUBJECT, rois=ONE_ROI)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS, layers=[LAYER],
                            subjects=ONE_SUBJECT, rois=ONE_ROI)

    # The statistics are where evaluation.py looks them up, unchanged:
    # <decoder>/<layer>/<subject>/<roi>/model/.
    for key in ('y_mean', 'y_norm'):
        assert os.path.isfile(os.path.join(
            pipeline.layer_model_dir(decoder_dir, LAYER, 'sub-01', 'VC'),
            '%s.mat' % key))

    output = pipeline.run_evaluation(
        dataset, decoder_dir, decoded_dir, synthetic.write_test_features(dataset),
        layers=[LAYER], subjects=ONE_SUBJECT, rois=ONE_ROI)

    from evaluation import ResultsStore
    results = ResultsStore(output)
    for metric in ('profile_correlation', 'pattern_correlation',
                   'identification_accuracy'):
        value = results.get(layer=LAYER, subject='sub-01', roi='VC',
                            metric=metric)
        assert value is not None and np.asarray(value).size > 0, metric
