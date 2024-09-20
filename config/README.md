# Configuration File

The configuration file is used for decoding DNN (Deep Neural Network) features from fMRI data.
The ccomponents in the configuration file is hierarchically organized
Below is a breakdown of the key components and parameters:

Example: [deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml](https://github.com/KamitaniLab/feature-decoding/blob/main/config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml)

## Decoder

This section defines settings for training of feature decoders.
These settings are used by the decoder training script (e.g., `train_decoder_fastl2lir.py`).

1. General
  - This specifies general settings about the decoders.
  - `decoder.name`: Specifies the name of the decoder model being used, in this case, "deeprecon_fmriprep_pyfastl2lir_alpha100_allunits".
  - `decoder.path`: Defines the path where the decoder data is located. The path can include placeholders like `${decoder.name}` and `${decoder.features.name}` to dynamically set paths based on the feature name.
  - `decoder.parameters`:
    - `alpha`: A regularization parameter set to 100.
    - `chunk_axis`: Indicates that chunking is done along axis 1.
2. fMRI data
  - This specifies fMRI data used for the decoder training.
  - `decoder.fmri.name`: Refers to the dataset name, here "ImageNetTraining_fmriprep_volume_native".
  - `decoder.fmri.subjects`: Lists the subjects included in the analysis. The corresponding path to the subject’s fMRI data is specified.
  - `decoder.fmri.rois`: Defines Regions of Interest (ROIs) in the brain and the number of voxels being used for the decoders.
  - `decoder.fmri.label_key`: Specifies "stimulus_name" as the key used to label the data in alignment with the stimuli presented to the subject during the experiment.
3. Features
  - This specifies features used for the decoder training.
  - `decoder.features.name`: Refers to the pre-trained DNN model used.
  - `decoder.features.paths`: Provides the path to the feature data.
  - `decoder.features.layers`: Specifies the layers of the DNN from which features will be extracted.

## Decoded features

This section defines settings about decoded (predicted) features.
These setting are used by the feature prediction (`predict_feature.py`) and evaluation (`evaluation.py`) scripts.

1. General
  - This specifies general settings about the decoded (predicted) features.
  - `decoded_feature.name`: Refers to the name of the decoded feature set ("train_deeprecon_rep5_test_ImageNetTest_fmriprep_pyfastl2lir_alpha100_allunits").
  - `decoded_feature.path`: Provides the path where the decoded features will be stored.
  - `decoded_feature.parameters`:
    - `average_sample`: Set to true, indicating that the decoded features will be averaged across samples.
2. Decoders
  - This specifies feature decoders used for the prediction of features.
  - `decoded_feature.decoder.name`: Specifies the name and path of the decoder used for the feature prediction.
3. fMRI data
  - This specifies fMRI data used for the prediction of features.
  - `decoded_feature.fmri.name`: Refers to the test dataset, "ImageNetTest_fmriprep_volume_native", which is used for decoding.
  - `decoded_feature.fmri.subjects`: Lists the fMRI data from each subject, along with the path to the data.
  - `decoded_feature.fmri.rois`: As in the earlier section, the visual cortex (VC) is selected for analysis.
  - `decoded_feature.fmri.label_key`: Defines stimulus_name as the labeling key.
  - `decoded_feature.fmri.exclude_labels`: Specifies labels excluded from the feature prediction. The feature prediction code ignores the fMRI samples corresponding to the labels specified here.
4. Features
  - This specifies the prediction target faetures. This is only used for the evaluation of decoding accuracies and not used in the prediction step (e.g., `predict_feature.py`).
  - `decoded_feature.features.name`: The name of DNN model (caffe/VGG19) used here for feature extraction.
  - `decoded_feature.features.paths`: Provides the path to the feature data for the test set.
  - `decoded_feature.features.layers`: Specifies the layers for which features will be decoded.
