# Feature decoding

This repository provides scripts of deep neural network (DNN) feature decoding from fMRI brain activities, originally proposed by [Horikawa & Kamitani (2017)](https://www.nature.com/articles/ncomms15037) and employed in DNN-based image reconstruction methods of [Shen et al. (2019)](http://dx.doi.org/10.1371/journal.pcbi.1006633) as well as recent studies in Kamitani lab.

## Usage

### Environemnt setup

Please setup Python environemnt where packages in [requirements.txt](requirements.txt) are installed.

```shell
# Using venv
$ python -m venv .venv
$ . .venv/bin/activate
$ pip install -r requirements.txt
```

### Data setup

TBA

### Decoding with PyFastL2LiR

- Training: `train_decoder_fastl2lir.py`
- Test (prediction): `predict_feature_fastl2lir.py`
- Evaluation: `evaluation.py`
- Example config file: [deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml](config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml)

```shell
# Training of decoding models
$ python train_decoder_fastl2lir.py config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml

# Prediction of DNN features
$ python predict_feature_fastl2lir.py config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml

# Evaluation
$ python evaluation.py config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml
```

### Decoding with generic regression models

- Training: `train_decoder_sklearn_ridge.py` (example for scikit-learn Ridge regression)
- Test (prediction): `predict_feature.py`
- Evaluation: `evaluation.py`
- Example config file: [deeprecon_sklearn_ridge_alpha100_vgg19_allunits](config/deeprecon_sklearn_ridge_alpha100_vgg19_allunits.yaml)

```shell
# Training of decoding models
$ python train_decoder_sklearn_ridge.py config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml

# Prediction of DNN features
$ python preeict_feature.py config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml

# Evaluation
$ python evaluation.py config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml
```

### Cross-validation feature decoding

- Training: `cv_train_decoder_fastl2lir.py` (example for scikit-learn Ridge regression)
- Test (prediction): `cv_predict_feature_fastl2lir.py`
- Evaluation: `cv_evaluation.py`
- Example config file: [deeprecon_cv_pyfastl2lir_alpha100_vgg19_allunits](config/deeprecon_cv_pyfastl2lir_alpha100_vgg19_allunits.yaml)

```shell
# Training of decoding models
$ python cv_train_decoder_fastl2lir.py config/deeprecon_cv_pyfastl2lir_alpha100_vgg19_allunits.yaml

# Prediction of DNN features
$ python cv_predict_feature_fastl2lir.py config/deeprecon_cv_pyfastl2lir_alpha100_vgg19_allunits.yaml

# Evaluation
$ python cv_evaluation.py config/deeprecon_cv_pyfastl2lir_alpha100_vgg19_allunits.yaml
```

## References

- Horikawa and Kamitani (2017) Generic decoding of seen and imagined objects using hierarchical visual features. *Nature Communications* 8:15037. https://www.nature.com/articles/ncomms15037
- Shen, Horikawa, Majima, and Kamitani (2019) Deep image reconstruction from human brain activity. *PLOS Computational Biology*. https://doi.org/10.1371/journal.pcbi.1006633