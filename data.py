# data.py
"""
This file handles loading data into the stratified-NTF pipeline. 

The current setup is:
    list of data from different strata
    Each element of the list is a tensor of shape
        samples, rows, columns
"""

from typing import Any, Optional, Union

import numpy as np
import PIL
import torchvision

getSyntheticReturnType = tuple[
    list[list[np.ndarray]],  # v
    list[np.ndarray],  # w
    list[np.ndarray],  # h
    np.array,  # losses
]


def get_synthetic(
    strata: int,
    num_samples: list[int],
    strata_feature_rank: list[int],
    topics_rank: int,
    fixed_dimensions: list[int],
) -> getSyntheticReturnType:
    """Construct synthetic data from hyperparameter settings.
    
    Args:
        strata: number of strata
        num_samples: number of samples in each stratum
        strata_feature_rank: number of strata features for each stratum
        topics_rank: number of global topics
        fixed_dimensions: dimensions of fixed modes (all but first)
    
    Returns:
        Randomly initialized v, w, and h arrays which can be combined to create
        a synthetic data tensor.
    """
    
    v = [
        [np.random.random((strata_feature_rank[i], d)) for d in fixed_dimensions]
        for i in range(strata)
    ]
    w = [np.random.random((topics_rank, num_samples[i])) for i in range(strata)]

    h = [np.random.random((topics_rank, d)) for d in fixed_dimensions]

    return v, w, h


def get_mnist(
    strata_dict: dict[str, list[Any]],
    strata_size: int,
) -> list[np.ndarray]:
    """Returns a stratified version of a subset of the MNIST dataset from torchvision.
    Where each stratum contains disjoint flattened digits defined by the strata_dict
    and strata_size parameters.

    Args:
        strata_dict: A dictionary with strata name and a list of labels
            ex strata_dict = {"12": [1, 2], "23": [2, 3]}
        strata_size: number of samples in each class that go into a strata
            eg s1 has 3 classes and s2 has 2 classes, then s1 has 300 points and s2 has 200 points

    Returns:
        List of matrices containing the data for each strata.
            Rows of each matrix are flattened images.
    """
    data = torchvision.datasets.MNIST(root="./Datasets", train=True, download=True)

    # First separate by the labels
    class_indices = [np.where(data.targets.numpy() == i)[0] for i in range(10)]

    # Now get the subset indices
    strata_indices = []
    for s_ind, val in enumerate(strata_dict.values()):
        strata_indices.append(
            np.concatenate(
                [
                    class_indices[i][s_ind * strata_size : (s_ind + 1) * strata_size]
                    for i in val
                ]
            )
        )

    return [data.data.numpy()[s_inds, ...] for s_inds in strata_indices]


def image_to_numpy(fn) -> np.ndarray:
    """Converts an image filename to a numpy array."""
    return np.asarray(PIL.Image.open(fn))


def load_numeric_watermarks() -> np.ndarray:
    """Loads premade numeric watermarks for noisy watermarked MNIST experiment."""
    path = "watermarks/"
    return np.array(
        [
            image_to_numpy(path + "one.png"),
            image_to_numpy(path + "two.png"),
            image_to_numpy(path + "three.png"),
            image_to_numpy(path + "onelow.png"),
            image_to_numpy(path + "twolow.png"),
            image_to_numpy(path + "threelow.png"),
        ]
    )[:, :, :, 0]


def get_mnist_watermarked(
    strata_dict: dict[str, list[Any]],
    strata_size: int,
    watermark_list: list[Any],
) -> list[np.ndarray]:
    """Returns a stratified version of a subset of the MNIST dataset from torchvision.
    Where each stratum contains disjoint flattened digits defined by the strata_dict
    and strata_size parameters. Watermarks are added as in watermark_list. If
    watermark_list contains integers, watermarks will be horizontal white lines at
    the corresponding row.

    Args:
        strata_dict: A dictionary with strata name and a list of labels
            ex strata_dict = {"12": [1, 2], "23": [2, 3]}
        strata_size: number of samples in each class that go into a strata
            eg s1 has 3 classes and s2 has 2 classes, then s1 has 300 points and s2 has 200 points
        watermark_list: a list containing either arrays as watermarks or integers
            corresponding to row indices for horizontal watermark.

    Returns:
        List of matrices containing the data for each strata.
            Rows of each matrix are flattened images.
    """
    data = get_mnist(strata_dict, strata_size)
    n, l = data[0][0].shape
    for strata in range(len(strata_dict)):
        if isinstance(watermark_list[strata], int):
            row = watermark_list[strata]
            watermark_list[strata] = np.zeros((n, l))
            watermark_list[strata][row] = 255

        watermark_list[strata].reshape(1, n, l)

        data[strata] = np.clip(
            data[strata].astype(int) + watermark_list[strata], 0, 255
        )

    return data


if __name__ == "__main__":
    strata_dict = {"12": [1, 2], "23": [2, 3]}
    tmp = get_mnist_watermarked(strata_dict, 100, load_numeric_watermarks()[[0, 4]])
    print(tmp[0].shape)
    print(tmp[1].shape)
