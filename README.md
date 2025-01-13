# SpinPath

[![Continuous Integration](https://github.com/SBU-BMI/spinpath/actions/workflows/cli-test.yml/badge.svg)](https://github.com/SBU-BMI/spinpath/actions/workflows/cli-test.yml)
[![Version on PyPI](https://img.shields.io/pypi/v/spinpath.svg)](https://pypi.org/project/spinpath/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/spinpath)](https://pypi.org/project/spinpath/)

SpinPath is a command line tool to run pre-trained MIL models on whole slide images. It is the slide-level companion to [WSInfer](https://wsinfer.readthedocs.io/en/latest/), which provides patch-level classification.

> [!CAUTION]
> SpinPath is intended _only_ for research purposes.

# Install

SpinPath can be installed using `pip`. SpinPath will install PyTorch automatically
if it is not installed, but this may not install GPU-enabled PyTorch even if a GPU is available.
For this reason, _install PyTorch before installing SpinPath_.

## Install PyTorch first

Please see [PyTorch's installation instructions](https://pytorch.org/get-started/locally/)
for help installing PyTorch. The installation instructions differ based on your operating system
and choice of `pip` or `conda`. Thankfully, the instructions provided
by PyTorch also install the appropriate version of CUDA. We refrain from including code
examples of installation commands because these commands can change over time. Please
refer to [PyTorch's installation instructions](https://pytorch.org/get-started/locally/)
for the most up-to-date instructions.

You will need a new-enough driver for your NVIDIA GPU. Please see
[this version compatibility table](https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility)
for the minimum versions required for different CUDA versions.

To test whether PyTorch can detect your GPU, check that this code snippet prints `True`.

```
python -c 'import torch; print(torch.cuda.is_available())'
```

## Install SpinPath with pip

```
pip install spinpath
```

# Examples

> [!CAUTION]
> These models are intended _only_ for research purposes.

## With a model online

Jakub Kaczmarzyk has uploaded several pre-trained MIL models to HuggingFace for the community to explore. Over time, I (Jakub) hope that others may contribute MIL models too. If you are interested in this, please feel free to email me at jakub.kaczmarzyk at stonybrookmedicine dot edu.

The models are available at https://huggingface.co/kaczmarj

### TP53 mutation prediction

```
spinpath run -m kaczmarj/pancancer-tp53-mut.tcga -i slide.svs
```

### Cancer tissue classification

```
spinpath run -m kaczmarj/pancancer-tissue-classifier.tcga -i slide.svs
```

### Metastasis prediction in axillary lymph nodes

```
spinpath run -m kaczmarj/breast-lymph-nodes-metastasis.camelyon16 -i slide.svs
```

### Survival prediction in GBM-LGG

```
spinpath run -m kaczmarj/gbmlgg-survival-porpoise.tcga -i slide.svs
```

### Survival prediction in kidney renal papillary cell carcinoma

```
spinpath run -m kaczmarj/kirp-survival-porpoise.tcga -i slide.svs
```


## With a local (potentially private) model

You can use SpinPath with a local MIL model. The model must be saved to TorchScript format, and a model configuration file must also be written.

Here is an example of a configuration JSON file:

```json
{
    "spec_version": "1.0",
    "type": "abmil",
    "patch_size_um": 128,
    "feature_extractor": "ctranspath",
    "num_classes": 2,
    "class_names": [
        "wildtype",
        "mutant"
    ]
}
```

There is a JSON schema in `spinpath/schemas/model-config.schema.json` for reference.

Once you have the model in TorchScript format and the configuration JSON file, you can run the model on slides. For example:

```
spinpath runlocal -m model.pt -c model.config.json \
    -i slides/TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.svs
```

# How it works from 30,000 feet

The pipeline for attention-based MIL methods is rather standardized. Here are the steps that SpinPath takes. In the future, we would like to incorporate inference using graph-based methods, so this workflow will likely have to be modified.

1. Segment the tissue in the image.
2. Create patches of the tissue regions.
3. Run a feature extractor on these patches.
4. Run the pre-trained model on the extracted features.
5. Save the results of the extracted features.

SpinPath caches steps 1, 2, and 3, as those can be reused among MIL models. Step 3 (feature extraction) is often the bottleneck of the workflow, and reusing extracted features can reduce runtime considerably.

# Developers

Clone and install `spinpath`:

Clone the repository and make a virtual environment for it. Then install the dependencies, with `dev` extras.

```
pip install -e .[dev]
```

Configure `pre-commit` to run the formatter before commits happen.

```
pre-commit install
```
