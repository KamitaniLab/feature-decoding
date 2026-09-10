"""Reading decoders saved by the direct (pre-factorization) implementation.

New training runs write the factorized format, but large decoders produced by
the previous implementation may already exist, so ``predict_feature.py`` keeps a
compatibility path.  The format is detected from the presence of
``factorized.yaml``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ridge_factorization import is_factorized_model_dir
from tests.helpers import legacy_ridge, pipeline

ALPHA = 100
CHUNK_AXIS = 1


def build_legacy_decoder(dataset, decoder_dir, layers=None,
                         subjects=('sub-01',), rois=('VC',),
                         chunk_axis=CHUNK_AXIS):
    """Train with the reference implementation and save the legacy layout."""
    trained = {}
    for layer in (layers or dataset.layers):
        for subject in subjects:
            for roi in rois:
                brain_labels = dataset.labels(subject, 'train')
                feat_labels = list(np.unique(brain_labels))
                entry = legacy_ridge.legacy_train(
                    dataset.brain(subject, roi, 'train'), brain_labels,
                    dataset.feature_matrix(layer, feat_labels), feat_labels,
                    alpha=ALPHA, chunk_axis=chunk_axis)
                legacy_ridge.write_legacy_decoder(
                    pipeline.layer_model_dir(decoder_dir, layer, subject, roi),
                    entry)
                trained[(layer, subject, roi)] = entry
    return trained


def test_format_detection(dataset, tmp_path):
    factorized_dir = str(tmp_path / 'factorized')
    pipeline.run_training(dataset, factorized_dir, alpha=ALPHA)
    assert is_factorized_model_dir(
        pipeline.brain_model_dir(factorized_dir, 'sub-01', 'VC'))

    legacy_dir = str(tmp_path / 'legacy')
    build_legacy_decoder(dataset, legacy_dir, layers=['fc_like'])
    assert not is_factorized_model_dir(
        pipeline.brain_model_dir(legacy_dir, 'sub-01', 'VC'))
    assert not is_factorized_model_dir(
        pipeline.layer_model_dir(legacy_dir, 'fc_like', 'sub-01', 'VC'))


@pytest.mark.parametrize('layer', ['fc_like', 'conv_like'])
def test_legacy_decoder_predictions_are_unchanged(dataset, tmp_path, layer):
    """A legacy decoder still decodes exactly as it used to, chunked or not."""
    decoder_dir = str(tmp_path / 'legacy_decoders')
    decoded_dir = str(tmp_path / 'legacy_decoded')

    trained = build_legacy_decoder(dataset, decoder_dir, layers=[layer])

    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS, layers=[layer],
                            subjects=['sub-01'], rois={'VC': 'ROI_VC = 1'})

    test_brain, test_labels = legacy_ridge.average_test_brain(
        dataset.brain('sub-01', 'VC', 'test'), dataset.labels('sub-01', 'test'))
    expected = legacy_ridge.legacy_predict(trained[(layer, 'sub-01', 'VC')],
                                           test_brain)
    actual = pipeline.read_decoded_features(decoded_dir, layer, 'sub-01', 'VC',
                                            test_labels)

    np.testing.assert_allclose(actual, expected.astype(np.float32),
                               rtol=1e-5, atol=1e-6)


def test_legacy_decoder_needs_no_training_features(dataset, tmp_path):
    """The legacy path is self-contained: it must not require ``F`` on disk."""
    import predict_feature

    decoder_dir = str(tmp_path / 'legacy_decoders')
    decoded_dir = str(tmp_path / 'legacy_decoded')
    build_legacy_decoder(dataset, decoder_dir, layers=['fc_like'])

    predict_feature.featdec_predict(
        {'sub-01': dataset.test_fmri['sub-01']}, decoder_dir,
        output_dir=decoded_dir,
        rois={'VC': 'ROI_VC = 1'},
        label_key=dataset.label_key,
        layers=['fc_like'],
        average_sample=True,
        chunk_axis=CHUNK_AXIS,
        training_features_paths=None,
        analysis_name='legacy_without_features')

    names = pipeline.decoded_feature_labels(decoded_dir, 'fc_like', 'sub-01',
                                            'VC')
    assert names == dataset.unique_test_labels


def test_factorized_decoder_requires_training_features(dataset, tmp_path):
    """A clear error beats a confusing one -- and it must leave no output.

    The check runs before the output directory is created, so the failed run
    cannot make the next, correct one print "already done" and skip.
    """
    import os

    import predict_feature

    decoder_dir = str(tmp_path / 'decoders')
    decoded_dir = str(tmp_path / 'decoded')
    pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)

    def predict(training_features_paths, analysis_name):
        return predict_feature.featdec_predict(
            {'sub-01': dataset.test_fmri['sub-01']}, decoder_dir,
            output_dir=decoded_dir,
            rois={'VC': 'ROI_VC = 1'},
            label_key=dataset.label_key,
            layers=['fc_like'],
            average_sample=True,
            chunk_axis=CHUNK_AXIS,
            training_features_paths=training_features_paths,
            analysis_name=analysis_name)

    with pytest.raises(ValueError, match='training_features_paths'):
        predict(None, 'factorized_without_features')

    assert not os.path.exists(os.path.join(decoded_dir, 'fc_like', 'sub-01',
                                           'VC'))

    predict([dataset.train_features_dir], 'factorized_with_features')

    assert pipeline.decoded_feature_labels(
        decoded_dir, 'fc_like', 'sub-01', 'VC') == dataset.unique_test_labels


def test_legacy_prediction_still_copies_the_feature_index(dataset, tmp_path):
    """The pre-factorization contract: given an index, it lands in the output.

    The direct implementation copied ``feature_index_file`` as given, and a
    legacy decoder is predicted without training features, so the copy cannot
    depend on them.
    """
    import os

    from tests.helpers import synthetic

    decoder_dir = str(tmp_path / 'legacy_decoders')
    decoded_dir = str(tmp_path / 'legacy_decoded')
    build_legacy_decoder(dataset, decoder_dir, layers=['fc_like'])

    # Written into the working directory, which is what the old code resolved
    # against (conftest chdirs into the test's own tmp_path).
    synthetic.write_feature_index('index_local.mat', {'fc_like': [0, 2]})

    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=CHUNK_AXIS, layers=['fc_like'],
                            subjects=['sub-01'], rois={'VC': 'ROI_VC = 1'},
                            features_paths=[],
                            feature_index_file='index_local.mat',
                            analysis_name='legacy_with_index')

    assert os.path.isfile(os.path.join(decoded_dir, 'feature_index.mat'))


def test_legacy_and_factorized_decoders_agree(dataset, tmp_path):
    """The compatibility path and the new path give the same numbers."""
    layer = 'conv_like'

    legacy_dir = str(tmp_path / 'legacy_decoders')
    legacy_decoded = str(tmp_path / 'legacy_decoded')
    build_legacy_decoder(dataset, legacy_dir, layers=[layer])
    pipeline.run_prediction(dataset, legacy_dir, legacy_decoded,
                            chunk_axis=CHUNK_AXIS, layers=[layer],
                            subjects=['sub-01'], rois={'VC': 'ROI_VC = 1'},
                            analysis_name='predict_legacy')
    legacy_pred = pipeline.read_decoded_features(
        legacy_decoded, layer, 'sub-01', 'VC', dataset.unique_test_labels)

    new_dir = str(tmp_path / 'new_decoders')
    new_decoded = str(tmp_path / 'new_decoded')
    pipeline.run_training(dataset, new_dir, alpha=ALPHA,
                          subjects=['sub-01'], rois={'VC': 'ROI_VC = 1'},
                          analysis_name='train_new')
    pipeline.run_prediction(dataset, new_dir, new_decoded,
                            chunk_axis=CHUNK_AXIS, layers=[layer],
                            subjects=['sub-01'], rois={'VC': 'ROI_VC = 1'},
                            analysis_name='predict_new')
    new_pred = pipeline.read_decoded_features(
        new_decoded, layer, 'sub-01', 'VC', dataset.unique_test_labels)

    np.testing.assert_allclose(new_pred, legacy_pred, rtol=1e-4, atol=1e-5)
