# Stratified-NTF

This is the repo for the Stratified-NTF project.

## Installation

To install the conda environment, run `conda env create -f environment.yml`.

The environment can be activated with `conda activate stratified_ntf`.

## Usage

Experiments can be run by running each of the following files:

`20newsgroups.py`

`olivetti_faces.py`

`noisy_mnist_watermarked.py`

Plots and logs will be saved in the `experiments/` folder under a timestamp corresponding to when the file was run. Some information will be displayed in the terminal.

### Additional information

`watermarks/` : Contains text watermarks for noisy watermarked MNIST experiment.

`data.py` : Contains data loading functions.

`plotting.py`: Contains plotting functions and driver code for running experiments.

`regularization.py`: Contains additional terms to perform regularized multiplicative updates.

`stratified_nmf.py`: Contains code for the multiplicative updates and a function for running Stratified NMF. From `https://github.com/chapman20j/Stratified-NMF`.

`stratified_ntf.py`: Contains code for the multiplicative updates and a function for running Stratified NTF.

`tensorly_ntf.py`: Contains code for classical NTF adapted from Tensorly's `non_negative_parafac` implementation.

`utils.py`: Contains utility functions for computation.