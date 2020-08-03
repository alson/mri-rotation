# MRI rotation experiment

## Introduction

Hypothetical use case: What if MRI scans were printed on circular transparancies without any markings, and someone dropped an entire stack of transparancies. What would be an algorithm that could rotate all the transparancies back to their original orientation without needing any markings.

This is an experiment in training a deep neural network to learn to identify the correct orientation of randomly rotated head MRI slices.

## How to use

This is based on [Victor Huang's PyTorch template project](https://github.com/victoresque/pytorch-template).

This project uses [Poetry](https://python-poetry.org/) to manage dependencies and virtual environments. Use `poetry install` to set up a virtual environment with all dependencies installed. Then use `poetry shell` or `poetry run` to run commands inside this environment. All commands below should be run within the virtual environment created by Poetry from the project root.

### Data

This uses data from the [Alzheimer’s Disease Neuroimaging Initiative](http://adni.loni.usc.edu/). See the website for details about how to get access to the data, and if you qualify.

Now create the rotated scans by in a larger space (square with sides that are the diagonal of the original image) in 1 degree increments, and store this label as folder name in the current folder:

`find ../ADNI -iname \*.nii|while read f; do echo $f; python ./nfi_rotate.py "$f";done`

Create a train / test split (with the default 0.2 test fraction):

`python train_test_split.py data data/train data/test`

### Training the model

Train the network using the PyTorch template script:

`python train.py -c config.json`

### Visualizing training with Tensorboard

You may need to manually install setuptools >= 49.0.0 through `pip` due to (this poetry bug / missing feature)[https://github.com/python-poetry/poetry/issues/1584]. Then run Tensorboard per the PyTorch template README:

`tensorboard --logdir saved/log/`
