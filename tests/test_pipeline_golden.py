"""End-to-end regression test against the committed golden fixtures.

The fixtures in ``tests/data/golden/`` were produced by the direct
(pre-factorization) sklearn Ridge implementation; regenerate them only with
``tests/generate_golden.py`` and only when the numerical output is *meant* to
change.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.conftest import GOLDEN_DIR
from tests.helpers import pipeline

GOLDEN_FILE = os.path.join(GOLDEN_DIR, 'sklearn_ridge_pipeline.npz')


@pytest.fixture(scope='module')
def golden():
    with np.load(GOLDEN_FILE) as f:
        yield f


def test_golden_fixture_is_present(golden):
    assert len(golden.files) > 0


def test_pipeline_reproduces_golden(dataset, decoder_dir, decoded_dir, golden):
    alpha = int(golden['_alpha'])
    chunk_axis = int(golden['_chunk_axis'])

    pipeline.run_training(dataset, decoder_dir, alpha=alpha,
                          chunk_axis=chunk_axis)
    pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                            chunk_axis=chunk_axis)

    test_labels = [str(x) for x in golden['_test_labels']]

    for layer in [str(x) for x in golden['_layers']]:
        for subject in [str(x) for x in golden['_subjects']]:
            for roi in [str(x) for x in golden['_rois']]:
                prefix = '%s|%s|%s|' % (layer, subject, roi)

                actual = pipeline.read_decoded_features(
                    decoded_dir, layer, subject, roi, test_labels)
                np.testing.assert_allclose(
                    actual, golden[prefix + 'pred'], rtol=1e-4, atol=1e-5,
                    err_msg='decoded features drifted for %s/%s/%s'
                            % (layer, subject, roi))

                saved = pipeline.read_norm_params(decoder_dir, layer, subject,
                                                  roi)
                for key in pipeline.NORM_KEYS:
                    np.testing.assert_allclose(
                        saved[key], golden[prefix + key],
                        rtol=1e-6, atol=1e-7,
                        err_msg='%s drifted for %s/%s/%s'
                                % (key, layer, subject, roi))
