"""Regenerate the golden regression fixtures for the sklearn Ridge decoder.

The fixtures record the normalization parameters and decoded features produced
by ``train_decoder_sklearn_ridge.py`` + ``predict_feature.py`` on a fixed-seed
synthetic dataset.  They were generated from the *direct* (pre-factorization)
implementation and are what the factorized implementation must reproduce.

Only run this when the expected numerical output is intended to change, and say
so explicitly in the commit message.

Note that regeneration is not bit-for-bit stable across environments: the
pipeline runs in ``float32`` and the Ridge solve's summation order depends on
how BLAS partitions the work, which moves the last bits (observed ~5e-8
relative). That is why the tests compare against these fixtures with a
tolerance rather than exactly, and why regenerating them without an intended
behavior change only adds noise.

    uv run python -m tests.generate_golden
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

from tests.helpers import pipeline, synthetic

ALPHA = 100
CHUNK_AXIS = 1
SEED = 0
GOLDEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data', 'golden', 'sklearn_ridge_pipeline.npz')


def build(work_dir: str) -> dict:
    """Run the full pipeline in ``work_dir`` and collect everything of interest."""
    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        dataset = synthetic.make_dataset(os.path.join(work_dir, 'data'),
                                         seed=SEED)
        decoder_dir = os.path.join(work_dir, 'feature_decoders')
        decoded_dir = os.path.join(work_dir, 'decoded_features')

        pipeline.run_training(dataset, decoder_dir, alpha=ALPHA)
        pipeline.run_prediction(dataset, decoder_dir, decoded_dir,
                                chunk_axis=CHUNK_AXIS)

        golden = {
            '_alpha': np.array(ALPHA),
            '_chunk_axis': np.array(CHUNK_AXIS),
            '_seed': np.array(SEED),
            '_layers': np.array(dataset.layers),
            '_subjects': np.array(sorted(dataset.train_fmri)),
            '_rois': np.array(sorted(dataset.rois)),
            '_test_labels': np.array(dataset.unique_test_labels),
        }
        for layer in dataset.layers:
            for subject in sorted(dataset.train_fmri):
                for roi in sorted(dataset.rois):
                    prefix = '%s|%s|%s|' % (layer, subject, roi)
                    values = pipeline.read_norm_params(decoder_dir, subject,
                                                       roi)
                    values.update(pipeline.read_feature_statistics(
                        decoder_dir, layer, subject, roi))
                    for key, value in values.items():
                        golden[prefix + key] = value
                    golden[prefix + 'pred'] = pipeline.read_decoded_features(
                        decoded_dir, layer, subject, roi,
                        dataset.unique_test_labels)
        return golden
    finally:
        os.chdir(cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output', default=GOLDEN_FILE)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as work_dir:
        golden = build(work_dir)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, **golden)
    print('Saved %s (%d arrays)' % (args.output, len(golden)))


if __name__ == '__main__':
    main()
