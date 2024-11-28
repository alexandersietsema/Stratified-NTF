# regularization.py
"""
This file contains the regularization functions for the stratified NTF loss. 
The convention for each function is
reg(*inputs) = reg_numerator, reg_denominator

The loss regularization is just the additional loss
"""

import numpy as np


# * Utility functions
def get_reg_update_fn(name: str):
    if name == "none":
        return no_regularization_update
    elif name == "tv":
        return tv_regularization_update
    else:
        raise ValueError(f"Unknown regularization function: {name}")


def get_reg_loss_fn(name: str):
    if name == "none":
        return no_regularization_loss
    elif name == "tv":
        return tv_regularization_loss
    else:
        raise ValueError(f"Unknown regularization function: {name}")


# * Loss functions
def no_regularization_loss(x: np.ndarray):
    return 0


# TODO: Check this
def tv_regularization_loss(x: np.ndarray):
    """Total variation regularization."""
    return np.sum(np.abs(np.diff(x)))


# * Update functions


def no_regularization_update(x: np.ndarray):
    return 0, 0


# TODO: Check this
def tv_regularization_update(x: np.ndarray):
    """Total variation regularization."""
    tmp = np.sign(np.diff(x, axis=-1))
    m, d = tmp.shape
    out = np.zeros((m, d + 1))
    out[:, 1:-1] = -tmp[:, 1:] + tmp[:, :-1]
    out[:, 0] = -tmp[:, 0]
    out[:, -1] = tmp[:, -1]
    num = -np.minimum(out, 0)
    den = np.maximum(out, 0)
    return num, den
