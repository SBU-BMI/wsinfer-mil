# WSInfer Multiple Instance Learning (MIL)

WSInfer MIL is a command line tool to run pre-trained MIL models on whole slide images.

# Install

```
pip install wsinfer-mil
```

# Example

```
wsinfer-mil runlocal -m model.pt -c model.config.json \
    -i slides/TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F.svs
```

# How it works

1. Segment the tissue in the image.
2. Create patches of the tissue regions.
3. Run a feature extractor on these patches.
4. Run the pre-trained model on the extracted features.
5. Save the results of the extracted features.

This code includes caching, so feature extractors are not run more than they have to be.

# Developers

Clone and install `wsinfer-mil`:

Clone the repository and make a virtual environment for it. Then install the dependencies, with `dev` extras.

```
pip install -e .[dev]
```

Configure `pre-commit` to run the formatter before commits happen.

```
pre-commit install
```
