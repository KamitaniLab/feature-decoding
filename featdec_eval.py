'''Feature decoding evaluation.'''


import argparse
from itertools import product
import os
import re

from bdpy.dataform import Features, DecodedFeatures, SQLite3KeyValueStore
from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification
import hdf5storage
import numpy as np
import pandas as pd
import yaml


# Main #######################################################################

class ResultsStore(SQLite3KeyValueStore):
    """Results store for feature decoding evaluation."""
    pass


def featdec_eval(
        decoded_feature_dir,
        true_feature_dir,
        output_file='./evaluation.db',
        subjects=None,
        rois=None,
        features=None,
        feature_index_file=None,
        feature_decoder_dir=None,
        single_trial=False
):
    '''Evaluation of feature decoding.

    Input:

    - deocded_feature_dir
    - true_feature_dir

    Output:

    - output_file

    Parameters:

    TBA
    '''

    # Display information
    print('Subjects: {}'.format(subjects))
    print('ROIs:     {}'.format(rois))
    print('')
    print('Decoded features: {}'.format(decoded_feature_dir))
    print('')
    print('True features (Test): {}'.format(true_feature_dir))
    print('')
    print('Layers: {}'.format(features))
    print('')
    if feature_index_file is not None:
        print('Feature index: {}'.format(feature_index_file))
        print('')

    # Loading data ###########################################################

    # True features
    if feature_index_file is not None:
        features_test = Features(true_feature_dir, feature_index=feature_index_file)
    else:
        features_test = Features(true_feature_dir)

    # Decoded features
    decoded_features = DecodedFeatures(decoded_feature_dir)

    # Metrics ################################################################
    metrics = ['profile_correlation', 'pattern_correlation', 'identification_accuracy']

    # Evaluating decoding performances #######################################

    if os.path.exists(output_file):
        print('Loading {}'.format(output_file))
        results_db = ResultsStore(output_file)
    else:
        print('Creating new evaluation result store')
        keys = ["layer", "subject", "roi", "metric"]
        results_db = ResultsStore(output_file, keys=keys)

    for layer in features:
        print('Layer: {}'.format(layer))
        
        true_y = features_test.get(layer=layer)
        true_labels = features_test.labels

        for subject, roi in product(subjects, rois):
            print('Subject: {} - ROI: {}'.format(subject, roi))

            # Check if the evaluation is already done
            exists = True
            for metric in metrics:
                exists = exists and results_db.exists(layer=layer, subject=subject, roi=roi, metric=metric)
            if exists:
                print('Already done. Skipped.')
                continue

            # Load decoded features
            pred_y = decoded_features.get(layer=layer, subject=subject, roi=roi)
            pred_labels = decoded_features.selected_label

            if single_trial:
                pred_labels = [re.match('sample\d*-(.*)', x).group(1) for x in pred_labels]

            if not np.array_equal(pred_labels, true_labels):
                y_index = [np.where(np.array(true_labels) == x)[0][0] for x in pred_labels]
                true_y_sorted = true_y[y_index]
            else:
                true_y_sorted = true_y

            # Load Y mean and SD
            # Proposed by Ken Shirakawa. See https://github.com/KamitaniLab/brain-decoding-cookbook/issues/13.
            norm_param_dir = os.path.join(
                feature_decoder_dir,
                layer, subject, roi,
                'model'
            )

            train_y_mean = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_mean.mat'))['y_mean']
            train_y_std = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_norm.mat'))['y_norm']

            # Evaluation ---------------------------

            # Profile correlation
            if not results_db.exists(layer=layer, subject=subject, roi=roi, metric='profile_correlation'):
                results_db.set(layer=layer, subject=subject, roi=roi, metric='profile_correlation', value=[])
                r_prof = profile_correlation(pred_y, true_y_sorted)
                results_db.set(layer=layer, subject=subject, roi=roi, metric='profile_correlation', value=r_prof)

            # Pattern correlation
            if not results_db.exists(layer=layer, subject=subject, roi=roi, metric='pattern_correlation'):
                results_db.set(layer=layer, subject=subject, roi=roi, metric='pattern_correlation', value=[])
                r_patt = pattern_correlation(pred_y, true_y_sorted, mean=train_y_mean, std=train_y_std)
                results_db.set(layer=layer, subject=subject, roi=roi, metric='pattern_correlation', value=r_patt)

            # Pair-wise identification accuracy
            if not results_db.exists(layer=layer, subject=subject, roi=roi, metric='identification_accuracy'):
                results_db.set(layer=layer, subject=subject, roi=roi, metric='identification_accuracy', value=[])
                if single_trial:
                    ident = pairwise_identification(pred_y, true_y, single_trial=True, pred_labels=pred_labels, true_labels=true_labels)
                else:
                    ident = pairwise_identification(pred_y, true_y_sorted)
                results_db.set(layer=layer, subject=subject, roi=roi, metric='identification_accuracy', value=ident)

            # Summary
            print('Mean profile correlation:     {}'.format(np.nanmean(r_prof)))
            print('Mean pattern correlation:     {}'.format(np.nanmean(r_patt)))
            print('Mean identification accuracy: {}'.format(np.nanmean(ident)))

    print('All done')

    return output_file


# Entry point ################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'conf',
        type=str,
        help='analysis configuration file',
    )
    args = parser.parse_args()

    conf_file = args.conf

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    conf.update({
        '__filename__': os.path.splitext(os.path.basename(conf_file))[0]
    })

    decoded_feature_dir = os.path.join(
        conf['decoded feature dir'],
        conf['network']
    )

    if 'feature index file' in conf:
        feature_index_file = os.path.join(conf['training feature dir'][0], conf['network'], conf['feature index file'])
    else:
        feature_index_file = None

    if 'test single trial' in conf:
        single_trial = conf['test single trial']
    else:
        single_trial = False

    featdec_eval(
        decoded_feature_dir,
        os.path.join(conf['test feature dir'][0], conf['network']),
        output_file=os.path.join(decoded_feature_dir, 'evaluation.db'),
        subjects=list(conf['test fmri'].keys()),
        rois=list(conf['rois'].keys()),
        features=conf['layers'],
        feature_index_file=feature_index_file,
        feature_decoder_dir=os.path.join(
            conf['feature decoder dir'],
            conf['network']
        ),
        single_trial=single_trial
    )
