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

- Training: `featdec_fastl2lir_train.py`
- Test (prediction): `featdec_fastl2lir_predict.py`
- Evaluation: `featdec_eval.py`
- Example config file: [deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml](config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml)

```shell
# Training of decoding models
$ python featdec_fastl2lir_train.py config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml

# Prediction of DNN features
$ python featdec_fastl2lir_preeict.py config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml

# Evaluation
$ python featdec_eval.py config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml
```

### Decoding with generic regression models

Example config file: [deeprecon_sklearn_ridge_alpha100_vgg19_allunits](config/deeprecon_sklearn_ridge_alpha100_vgg19_allunits.yaml)

- Training: `featdec_sklearn_ridge_train.py` (example for scikit-learn Ridge regression)
- Test (prediction): `featdec_predict.py`
- Evaluation: `featdec_eval.py`
- Example config file: [deeprecon_sklearn_ridge_alpha100_vgg19_allunits](config/deeprecon_sklearn_ridge_alpha100_vgg19_allunits.yaml)

```shell
# Training of decoding models
$ python featdec_sklearn_ridge_train.py config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml

# Prediction of DNN features
$ python featdec_preeict.py config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml

# Evaluation
$ python featdec_eval.py config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml
```

### Cross-validation feature decoding

TBA

## References

- Horikawa and Kamitani (2017) Generic decoding of seen and imagined objects using hierarchical visual features. *Nature Communications* 8:15037. https://www.nature.com/articles/ncomms15037
- Shen, Horikawa, Majima, and Kamitani (2019) Deep image reconstruction from human brain activity. *PLOS Computational Biology*. https://doi.org/10.1371/journal.pcbi.1006633