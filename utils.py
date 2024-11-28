# utils.py
"""
This file contains utility functions that will simplify the update calculations. 
"""

import numpy as np
from icecream import ic

def multiply(array_list: list[np.ndarray], shape: tuple = None) -> np.ndarray:
    """Performs entrywise multiplication on a list of arrays."""
    if len(array_list) == 0:
        return np.ones(shape, dtype=np.float64)

    out = array_list[0].copy()
    for arr in array_list[1:]:
        out *= arr
    return out


def my_reshape(x: np.ndarray, ind: int, num_tensors: int) -> np.ndarray:
    """Reshapes a B x D tensor into a B x (1 x ... x 1 x D x 1 x .. x 1) tensor,
    where D is in the ind-th mode of the last num_tensors modes.
    
    Example: Let x be a 4 x 5 tensor.
        input: my_reshape(X, ind=4, num_tensors=6)
        output: tensor of shape 4 x (1 x 1 x 5 x 1 x 1 x 1)
        
    Args:
        x: mode-2 tensor to be reshaped
        ind: index of second mode of x in output
        num_tensors: total number of modes after first mode.
        
    Returns:
        Reshaped x tensor
    """

    B, D = x.shape
    dimension_tuple = (B, *((1,) * ind), D, *((1,) * (num_tensors - ind - 1)))
    return x.reshape(dimension_tuple)


def batch_tensor_product(tensor_list: list[np.ndarray]) -> np.ndarray:
    """
    Takes a list of mode-2 tensors with common first dimension and computes
    the outer product along the last dimension.

    Example: tensor_list is a list of tensors with shape B x D1, B x D2, ..., B x Dn
        input: batch_tensor_product(tensor_list)
        output: tensor of shape B x D1 x D2 x ... x Dn
        
    Args:
        tensor_list: list of two-dimensional arrays with common first dimension
    
    Returns:
        product tensor across the second dimension of each array in tensor_list
    """

    # Checks that all the beginning dimensions are the same
    assert all(
        tensor_list[i].shape[0] == tensor_list[0].shape[0]
        for i in range(1, len(tensor_list))
    )

    num_tensors = len(tensor_list)
    out = my_reshape(tensor_list[0], 0, num_tensors)
    
    # Compute outer product in each dimension
    for i in range(1, num_tensors):
        out = out * my_reshape(tensor_list[i], i, num_tensors)

    return out


def compute_Bs(
    num_samples: int,
    v: list[list[np.ndarray]],
    w: list[np.ndarray],
    h: list[np.ndarray],
    s: int,
) -> list[np.ndarray]:
    """Computes Stratified-NTF approximation of data tensor as in Eqn. (4)."""
    strata_feature_rank = v[s][0].shape[0]
    Bs = batch_tensor_product([np.ones((strata_feature_rank, num_samples))] + v[s]).sum(
        0
    ) + batch_tensor_product([w[s]] + h).sum(0)

    return Bs


def display_all_shapes(
    A: list[np.ndarray], 
    v: list[list[np.ndarray]],
    w: list[np.ndarray],
    h: list[np.ndarray]
) -> None:
    """Displays the shape of Stratified-NTF parameters."""
    ic([a.shape for a in A])
    for s in range(len(v)):
        ic([x.shape for x in v[s]])
    ic([x.shape for x in w])
    ic([x.shape for x in h])


def assert_nonnegative(
    v: list[list[np.ndarray]],
    w: list[np.ndarray],
    h: list[np.ndarray]
) -> None:
    """Checks that Stratified-NTF parameters are non-negative."""
    
    # Check the v's
    for x in v:
        for y in x:
            assert np.all(y >= 0)

    # Check the w's
    for x in w:
        assert np.all(x >= 0)

    # Check the h's
    for x in h:
        assert np.all(x >= 0)

