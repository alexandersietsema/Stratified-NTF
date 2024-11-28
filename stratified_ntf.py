# stratified_ntf.py
"""
This file implements the Stratified NTF algorithm. 
"""

import numpy as np
from scipy.sparse import csr_array
from termcolor import cprint
from tqdm import tqdm

from regularization import get_reg_loss_fn, get_reg_update_fn
from utils import assert_nonnegative, batch_tensor_product, compute_Bs, multiply

StratifiedNTFReturnType = tuple[
    list[list[np.ndarray]],  # v
    list[np.ndarray],  # w
    list[np.ndarray],  # h
    np.array,  # losses
]

DEFAULT_TOL = 1e-9


def update_v(
    A: list[np.ndarray],
    v: list[list[np.ndarray]],
    w: list[np.ndarray],
    h: list[np.ndarray],
    tol: float = DEFAULT_TOL,
    reg_type: str = "none",
    reg: float = 1.0,
) -> np.ndarray:
    """Multiplicative update for V

    Args:
        A: List of length (strata) of ndarrays of shape (d_1(i), d_2, ..., d_modes)
        v: List of length (strata) containing a list of length (modes-1) or ndarrays of shape (rank,d_k) where k corresponds to the mode
        w: List of length (strata) containing ndarrays of shape (rank, d_1(i)).
        h: List of legth (modes-1) containing ndarrays of shape (rank,d_k) where k corresponds to the mode
        tol: Parameter to prevent numerical instabilities. Defaults to DEFAULT_TOL.
        reg_type: Type of regularization to use. Defaults to "none".
        reg: Scaling parameter for regularization. Defaults to 1.0.

    Returns:
        new values for V
    """
    strata = len(A)

    out = [[vvv.copy() for vvv in vv] for vv in v]
    num_samples = [x.shape[0] for x in A]
    modes = len(A[0].shape)

    for s in range(strata):

        # Compute reduced versions of A and B
        sum_A = A[s].sum(axis=0)

        for t in range(modes - 1):

            v_tensor = batch_tensor_product(
                [x.copy() if i != t else np.ones_like(x) for i, x in enumerate(v[s])]
            )
            sum_inds = tuple([i for i in range(1, modes) if i != t + 1])

            num = v_tensor * sum_A
            num = num.sum(axis=sum_inds)

            v_denom = num_samples[s] * (
                multiply(
                    [v[s][k] @ v[s][k].T for k in range(modes - 1) if k != t],
                    shape=(v[s][0].shape[0], v[s][0].shape[0]),
                )
                @ v[s][t]
            )
            wh_denom = multiply(
                [v[s][k] @ h[k].T for k in range(modes - 1) if k != t],
                shape=(v[s][0].shape[0], h[0].shape[0]),
            ) @ (h[t] * w[s].sum(1).reshape(-1, 1))

            denom = v_denom + wh_denom

            num = np.clip(num, a_min=tol, a_max=None)
            denom = np.clip(denom, a_min=tol, a_max=None)

            out[s][t] = v[s][t].copy() * num / (denom)

            # update the modes sequentially
            v[s][t] = out[s][t]
    return out


def update_w(
    A: list[np.ndarray],
    v: list[list[np.ndarray]],
    w: list[np.ndarray],
    h: list[np.ndarray],
    tol: float = DEFAULT_TOL,
    reg_type: str = "none",
    reg: float = 1.0,
) -> np.ndarray:
    """Multiplicative update for W

    Args:
        A: List of length (strata) of ndarrays of shape (d_1(i), d_2, ..., d_modes)
        v: List of length (strata) containing a list of length (modes-1) or ndarrays of shape (rank,d_k) where k corresponds to the mode
        w: List of length (strata) containing ndarrays of shape (rank, d_1(i)).
        h: List of legth (modes-1) containing ndarrays of shape (rank,d_k) where k corresponds to the mode
        tol: Parameter to prevent numerical instabilities. Defaults to DEFAULT_TOL.
        reg_type: Type of regularization to use. Defaults to "none".
        reg: Scaling parameter for regularization. Defaults to 1.0.

    Returns:
        new values for W
    """
    # NOTE: This has been confirmed
    strata = len(A)
    out = [None for _ in range(strata)]

    H = batch_tensor_product(h)
    num_samples = [x.shape[0] for x in A]
    modes = len(A[0].shape)

    for s in range(strata):

        num = H.reshape(H.shape[0], -1) @ np.transpose(A[s].reshape(A[s].shape[0], -1))

        v_denom = np.sum(multiply([h[k] @ v[s][k].T for k in range(modes - 1)]), axis=1)
        v_denom = np.repeat(v_denom.reshape(-1, 1), num_samples[s], axis=1)

        wh_denom = multiply([h[k] @ h[k].T for k in range(modes - 1)]) @ w[s]

        denom = v_denom + wh_denom

        num = np.clip(num, a_min=tol, a_max=None)
        denom = np.clip(denom, a_min=tol, a_max=None)

        out[s] = w[s] * num / (denom)

    return out


def update_h(
    A: list[np.ndarray],
    v: list[list[np.ndarray]],
    w: list[np.ndarray],
    h: list[np.ndarray],
    tol: float = DEFAULT_TOL,
    reg_type: str = "none",
    reg: float = 1.0,
) -> np.ndarray:
    """Multiplicative update for H

    Args:
        A: List of length (strata) of ndarrays of shape (d_1(i), d_2, ..., d_modes)
        v: List of length (strata) containing a list of length (modes-1) or ndarrays of shape (rank,d_k) where k corresponds to the mode
        w: List of length (strata) containing ndarrays of shape (rank, d_1(i)).
        h: List of legth (modes-1) containing ndarrays of shape (rank,d_k) where k corresponds to the mode
        tol: Parameter to prevent numerical instabilities. Defaults to DEFAULT_TOL.
        reg_type: Type of regularization to use. Defaults to "none".
        reg: Scaling parameter for regularization. Defaults to 1.0.

    Returns:
        new values for H
    """
    # NOTE: This has been confirmed
    strata = len(A)
    modes = len(A[0].shape)

    num_sum = [0 for _ in range(modes)]
    denom_sum = [0 for _ in range(modes)]

    out = [hh.copy() for hh in h]

    # * REG
    # NOTE: Just doing on first part of h for now
    reg_update_fn = get_reg_update_fn(reg_type)
    reg_num = dict()
    reg_den = dict()
    for i in range(2):
        tmp = reg_update_fn(h[i])
        reg_num[i] = tmp[0]
        reg_den[i] = tmp[1]

    for tt in range(modes - 1):

        for s in range(strata):

            t = tt + 1

            # Compute wxH and the sum indices
            wxH = batch_tensor_product(
                [w[s].copy()]
                + [x.copy() if i != t - 1 else np.ones_like(x) for i, x in enumerate(h)]
            )

            sum_indices = tuple([i for i in range(1, modes + 1) if i != t + 1])

            # compute numerator
            num = wxH * A[s]
            num = num.sum(axis=sum_indices)

            v_denom = (
                multiply(
                    [h[k] @ v[s][k].T for k in range(modes - 1) if k != tt],
                    shape=(h[0].shape[0], v[s][0].shape[0]),
                )
                @ v[s][tt]
            ) * w[s].sum(1).reshape((-1, 1))
            wh_denom = (
                (w[s] @ w[s].T)
                * multiply(
                    [h[k] @ h[k].T for k in range(modes - 1) if k != tt],
                    shape=(h[0].shape[0], h[0].shape[0]),
                )
            ) @ h[tt]

            denom = v_denom + wh_denom

            num = np.clip(num, a_min=tol, a_max=None)
            denom = np.clip(denom, a_min=tol, a_max=None)

            num_sum[t - 1] += num
            denom_sum[t - 1] += denom

        # after computing sum across all strata, update mode
        num, denom = num_sum[tt], denom_sum[tt]
        if tt in reg_num:
            num += reg * reg_num[tt]
            denom += reg * reg_den[tt]
            assert np.all(reg_num[tt] >= 0)
            assert np.all(reg_den[tt] >= 0)
        out[tt] = h[tt] * (num / denom)

        # new result is used immediately to perform sequential mode updates
        h[tt] = out[tt]

    return out


def loss(
    A: list[np.ndarray],
    v: list[list[np.ndarray]],
    w: list[np.ndarray],
    h: list[np.ndarray],
    reg_type: str = "none",
    reg: float = 1.0,
) -> float:
    """Loss function for stratified NTF

    Args:
        A: List of length (strata) of ndarrays of shape (d_1(i), d_2, ..., d_modes)
        v: List of length (strata) containing a list of length (modes-1) or ndarrays of shape (rank,d_k) where k corresponds to the mode
        w: List of length (strata) containing ndarrays of shape (rank, d_1(i)).
        h: List of legth (modes-1) containing ndarrays of shape (rank,d_k) where k corresponds to the mode
        tol: Parameter to prevent numerical instabilities. Defaults to DEFAULT_TOL.
        reg_type: Type of regularization to use. Defaults to "none".
        reg: Scaling parameter for regularization. Defaults to 1.0.

    Returns:
        Loss value
    """
    strata = len(A)
    out = 0.0
    num_samples = [x.shape[0] for x in A]

    for s in range(strata):

        Bs = compute_Bs(num_samples[s], v, w, h, s)
        out += np.linalg.norm(A[s] - Bs) ** 2

    out = out**0.5

    reg_loss_fn = get_reg_loss_fn(reg_type)
    for hh in h:
        out += reg * (reg_loss_fn(hh[0]) + reg_loss_fn(hh[1]))
    return out


def stratified_ntf(
    A: list[np.ndarray],
    strata_feature_rank: int,
    topics_rank: int,
    iters: int,
    v_scaling: int = 2,
    calculate_loss: bool = True,
    starting=None,
    disable: bool = False,
    tol=DEFAULT_TOL,
    reg_type: str = "none",
    reg: float = 1.0,
) -> StratifiedNTFReturnType:
    """Runs stratified NTF on the given data

    Args:
        A: List of data tensors
        strata_feature_rank: Rank for the strata features
        topics_rank: Rank for the topics
        iters: iterations to run
        v_scaling: Times to update v every iteration. Defaults to 2.
        calculate_loss: Whether to calculate loss. Defaults to True.
        starting: parameter initialization. Defaults to None.
        disable: whether to disable tqdm. Defaults to False.
        tol: Parameter to prevent numerical instabilites. Defaults to 1e-9.
        reg_type: Type of regularization to use. Defaults to "none".
        reg: Scaling parameter for regularization. Defaults to 1.0.

    Returns:
        Learned v, w, h, and loss array
    """

    strata = len(A)

    # check if strata_feature_rank is an int, allowing for np.ints and others
    # as well. If not, make sure that it is an iterable of ints.
    try:
        if int(strata_feature_rank) == strata_feature_rank:
            strata_feature_rank = [strata_feature_rank] * len(A)
    except:
        assert np.all([int(ss) == ss for ss in strata_feature_rank])

    # d2, d3, ...
    fixed_dimensions = A[0].shape[1:]

    # d1(s)
    num_samples = [x.shape[0] for x in A]

    if starting:
        v, w, h = starting
    else:
        v = [
            [np.random.random((strata_feature_rank[i], d)) for d in fixed_dimensions]
            for i in range(strata)
        ]
        w = [np.random.random((topics_rank, num_samples[i])) for i in range(strata)]

        h = [np.random.random((topics_rank, d)) for d in fixed_dimensions]

    # Keep track of loss array
    loss_array = np.zeros(iters)

    if isinstance(A[0], csr_array) and calculate_loss:
        cprint(
            "Warning: loss calculation decreases performance when using large, sparse matrices.",
            "yellow",
        )

    # Run S-NTF
    for i in tqdm(range(iters), disable=disable):

        # Calculate loss
        if calculate_loss:
            loss_array[i] = loss(A, v, w, h, reg_type, reg)

        # Update V
        for _ in range(v_scaling):
            v = update_v(A, v, w, h, tol, reg_type, reg)

        # Update W, H
        w = update_w(A, v, w, h, tol, reg_type, reg)
        h = update_h(A, v, w, h, tol, reg_type, reg)
        if reg_type == "tv":
            for i in range(len(h)):
                h[i] /= np.linalg.norm(h[i], axis=0)

    assert_nonnegative(v, w, h)

    return v, w, h, loss_array
