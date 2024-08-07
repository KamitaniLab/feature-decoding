'''Feature decoding (corss-validation) evaluation.'''


import argparse
from itertools import product
import os
import re

from bdpy.dataform import Features, DecodedFeatures, SQLite3KeyValueStore
from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification
import hdf5storage
import numpy as np
import yaml


# Main #######################################################################

class ResultsStore(SQLite3KeyValueStore):
    """Results store for feature decoding evaluation."""
    pass


def featdec_cv_eval(
        decoded_feature_dir,
        true_feature_dir,
        output_file_pooled='./evaluation.db',
        output_file_fold='./evaluation_fold.db',
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

    cv_folds = decoded_features.folds

    # Metrics ################################################################
    metrics = ['profile_correlation', 'pattern_correlation', 'identification_accuracy']
    pooled_operation = {
        "profile_correlation": "mean",
        "pattern_correlation": "concat",
        "identification_accuracy": "concat",
    }

    # Evaluating decoding performances #######################################

    if os.path.exists(output_file_fold):
        print('Loading {}'.format(output_file_fold))
        results_db = ResultsStore(output_file_fold)
    else:
        print('Creating new evaluation result store')
        keys = ["layer", "subject", "roi", "fold", "metric"]
        results_db = ResultsStore(output_file_fold, keys=keys)

    true_labels = features_test.labels

    for layer in features:
        print('Layer: {}'.format(layer))
        true_y = features_test.get_features(layer=layer)

        for subject, roi, fold in product(subjects, rois, cv_folds):
            print('Subject: {} - ROI: {} - Fold: {}'.format(subject, roi, fold))

            # Check if the evaluation is already done
            exists = True
            for metric in metrics:
                exists = exists and results_db.exists(layer=layer, subject=subject, roi=roi, fold=fold, metric=metric)
            if exists:
                print('Already done. Skipped.')
                continue

            pred_y = decoded_features.get(layer=layer, subject=subject, roi=roi, fold=fold)
            pred_labels = decoded_features.selected_label

            if single_trial:
                pred_labels = [re.match('trial_\d*-(.*)', x).group(1) for x in pred_labels]

            if not np.array_equal(pred_labels, true_labels):
                y_index = [np.where(np.array(true_labels) == x)[0][0] for x in pred_labels]
                true_y_sorted = true_y[y_index]
            else:
                true_y_sorted = true_y

            # Load Y mean and SD
            # Proposed by Ken Shirakawa. See https://github.com/KamitaniLab/brain-decoding-cookbook/issues/13.
            norm_param_dir = os.path.join(
                feature_decoder_dir,
                layer, subject, roi, fold,
                'model'
            )

            train_y_mean = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_mean.mat'))['y_mean']
            train_y_std = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_norm.mat'))['y_norm']

            # Evaluation ---------------------------

            # Profile correlation
            if not results_db.exists(layer=layer, subject=subject, roi=roi, fold=fold, metric='profile_correlation'):
                results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='profile_correlation', value=np.array([]))
                r_prof = profile_correlation(pred_y, true_y_sorted)
                results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='profile_correlation', value=r_prof)
                print('Mean profile correlation:     {}'.format(np.nanmean(r_prof)))

            # Pattern correlation
            if not results_db.exists(layer=layer, subject=subject, roi=roi, fold=fold, metric='pattern_correlation'):
                results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='pattern_correlation', value=np.array([]))
                r_patt = pattern_correlation(pred_y, true_y_sorted, mean=train_y_mean, std=train_y_std)
                results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='pattern_correlation', value=r_patt)
                print('Mean pattern correlation:     {}'.format(np.nanmean(r_patt)))

            # Pair-wise identification accuracy
            if not results_db.exists(layer=layer, subject=subject, roi=roi, fold=fold, metric='identification_accuracy'):
                results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='identification_accuracy', value=np.array([]))
                if single_trial:
                    ident = pairwise_identification(pred_y, true_y, single_trial=True, pred_labels=pred_labels, true_labels=true_labels)
                else:
                    ident = pairwise_identification(pred_y, true_y_sorted)
                results_db.set(layer=layer, subject=subject, roi=roi, fold=fold, metric='identification_accuracy', value=ident)
                print('Mean identification accuracy: {}'.format(np.nanmean(ident)))

    print('All fold done')

    # Pooled accuracy
    if os.path.exists(output_file_pooled):
        print('Loading {}'.format(output_file_pooled))
        pooled_db = ResultsStore(output_file_pooled)
    else:
        print('Creating new evaluation result store')
        keys = ["layer", "subject", "roi", "metric"]
        pooled_db = ResultsStore(output_file_pooled, keys=keys)

    done_all = True  # Flag indicating that all conditions have been pooled
    for layer, subject, roi, metric in product(features, subjects, rois, metrics):
        # Check if pooling is done
        if pooled_db.exists(layer=layer, subject=subject, roi=roi, metric=metric):
            continue
        pooled_db.set(layer=layer, subject=subject, roi=roi, metric=metric, value=np.array([]))

        # Check if all folds are complete
        done = True
        for fold in cv_folds:
            if not results_db.exists(layer=layer, subject=subject, roi=roi,
                                     fold=fold, metric=metric):
                done = False
                break

        # When all folds are complete, pool the results.
        if done:
            acc = []
            for fold in cv_folds:
                acc.append(results_db.get(layer=layer, subject=subject, roi=roi,
                                          fold=fold, metric=metric))
            if pooled_operation[metric] == "mean":
                acc = np.nanmean(acc, axis=0)
            elif pooled_operation[metric] == "concat":
                acc = np.hstack(acc)
            pooled_db.set(layer=layer, subject=subject, roi=roi,
                          metric=metric, value=acc)

        # If there are any unfinished conditions,
        # do not pool the results and set the done_all flag to False.
        else:
            pooled_db.delete(layer=layer, subject=subject, roi=roi, metric=metric)
            done_all = False
            continue

    if done_all:
        print('All pooling done.')
    else:
        print("Some pooling has not finished.")

    return output_file_pooled, output_file_fold


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

    if 'analysis name' in conf:
        analysis_name = conf['analysis name']
    else:
        analysis_name = ''

    decoded_feature_dir = os.path.join(
        conf['decoded feature dir'],
        analysis_name,
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

    featdec_cv_eval(
        decoded_feature_dir,
        os.path.join(conf['feature dir'][0], conf['network']),
        output_file_pooled=os.path.join(decoded_feature_dir, 'evaluation.db'),
        output_file_fold=os.path.join(decoded_feature_dir, 'evaluation_fold.db'),
        subjects=list(conf['fmri'].keys()),
        rois=list(conf['rois'].keys()),
        features=conf['layers'],
        feature_index_file=feature_index_file,
        feature_decoder_dir=os.path.join(
            conf['feature decoder dir'],
            analysis_name,
            conf['network']
        ),
        single_trial=single_trial
    )
