# plotting.py
"""
This file contains plotting functions and driver code for running experiments.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from typing import Iterable
#plt.rcParams["font.family"] = "Lato"
import pickle

from tensorly_ntf import non_negative_parafac
from stratified_nmf import stratified_nmf
from stratified_ntf import stratified_ntf
from utils import batch_tensor_product, compute_Bs

StratifiedNTFReturnType = tuple[
    list[list[np.ndarray]],  # v
    list[np.ndarray],  # w
    list[np.ndarray],  # h
    np.array,  # losses
]

RunExperimentsReturnType = list[
    list[np.ndarray, list[np.ndarray]], # classical ntf params
    list[float],  # classical ntf losses
    StratifiedNTFReturnType,  # stratified ntf params
    list[float],  # stratified ntf losses
    list[np.ndarray, list[np.ndarray], np.ndarray], # stratified nmf params
    list[float] # stratified nmf losses
]
def n_params(x):
    """Returns the number of elements in a list of arrays."""
    return sum(len(np.ravel(xxx)) for xx in x for xxx in xx)

def make_directory(dataset_name):
    """Creates a directory for new experiments."""
    experiment_name = f"{int(time.time())}"
    path = f"experiments/{dataset_name}/{experiment_name}"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def run_experiment(
    A: list[np.ndarray],
    strata_feature_rank: int = 10,
    topics_rank: int = 10,
    iters: int = 300,
    use_same_n_params: bool = True,
    do_classical_ntf: bool = True,
    do_stratified_ntf: bool = True,
    do_stratified_nmf: bool = True,
    path: str = "",
) -> RunExperimentsReturnType:
    """Run a comparison experiment.
    
    Args:
        A: Stratified data tensor
        strata_feature_rank: Number of strata features. Defaults to 10.
        topics_rank: Number of topics. Defaults to 10.
        iters: Number of iterations to run each method for. Defaults to 300.
        use_same_n_params: Whether to use the same number of total parameters for
            stratified ntf for the comparison methods. Prefers giving comparisons
            more parameters when needed. Defaults to True.
        do_classical_ntf: Whether to run classical ntf. Defaults to True.
        do_stratified_ntf: Whether to run stratified ntf. Defaults to True.
        do_stratified_nmf: Whether to run stratified nmf. Defaults to True.
        path: File path to save results to. Defaults to current working directory.
        
    Returns:
        List of parameters and losses for each method of form
        [ntf_params, losses_ntf, sntf_params, losses_sntf, snmf_params, losses_snmf]
    """
    (
        w_ntf,
        h_ntf,
        losses_ntf,
        v_sntf,
        w_sntf,
        h_sntf,
        losses_sntf,
        v_n,
        w_n,
        h_n,
        losses_snmf,
    ) = (None, None, None, None, None, None, None, None, None, None, None)

    try:
        if int(strata_feature_rank) == strata_feature_rank:
            strata_feature_rank = [strata_feature_rank] * len(A)
    except:
        assert np.all([int(ss) == ss for ss in strata_feature_rank])
        
    # create non-stratified data tensor for classical ntf
    Astack = np.concatenate(A)
    
    data_dims = Astack.shape[1:]
    num_samples = [len(AA) for AA in A]
    n_strata = len(A)

    ntf_params = (np.sum(num_samples) + np.sum(data_dims)) * topics_rank
    sntf_params = ntf_params + np.sum(data_dims) * np.sum(strata_feature_rank)

    ntf_rank = None
    snmf_rank = None
    snmf_params = None

    if do_classical_ntf:
        
        ntf_rank = topics_rank
        
        if use_same_n_params:
            # only update rank if classical ntf has fewer params than stratified_ntf
            assert sntf_params >= ntf_params

            # update ntf rank to make total number of parameters larger than stratified_ntf
            ntf_rank = int(np.ceil((sntf_params / ntf_params) * topics_rank))
            ntf_params = (np.sum(num_samples) + np.sum(data_dims)) * ntf_rank
            assert ntf_params >= sntf_params
        
        
        _, h_ntf, losses_ntf = non_negative_parafac(
            Astack, ntf_rank, iters, return_errors=True, init="random"
        )

        # reformat tensorly output shapes
        w_ntf = h_ntf[0].T
        h_ntf = [hh.T for hh in h_ntf[1:]]

        # w_ntf, h_ntf, losses_ntf = ntf(Astack, ntf_rank, iters)

        assert ntf_params == n_params(w_ntf) + n_params(h_ntf)
        print(f"number of parameters (ntf): {ntf_params}")

    if do_stratified_ntf:
        v_sntf, w_sntf, h_sntf, losses_sntf = stratified_ntf(
            A, strata_feature_rank, topics_rank, iters
        )
        assert sntf_params == n_params(v_sntf) + n_params(w_sntf) + n_params(h_sntf)
        print(f"number of parameters (sntf): {sntf_params}")

    if do_stratified_nmf:
        A_snmf = [AA.reshape(AA.shape[0], -1) for AA in A]
        
        # calculate the number of parameters for a given rank
        snmf_param_f = (
            lambda x: np.prod(data_dims) * (n_strata + x) + np.sum(num_samples) * x
        )
            
        # find smallest rank which gives a total number of parameters larger than stratified_ntf
        for x in range(1, topics_rank + 1):
            if snmf_param_f(x) >= sntf_params:
                break
        snmf_rank = x
        # snmf_rank = int(np.ceil((np.sum(data_dims)*topics_rank) / np.prod(data_dims)))
        # print(snmf_rank)
        v_n, w_n, h_n, losses_snmf = stratified_nmf(A_snmf, snmf_rank, iters)
        snmf_params = n_params(v_n) + n_params(w_n) + n_params(h_n)
        # print(f"number of parameters (snmf): {snmf_params} {snmf_rank}")

    # log experiment parameter settings in txt file
    with open(f"{path}/params.txt", "w") as f:

        out = f"""
        strata_feature_rank={strata_feature_rank},
        topics_rank={topics_rank},
        iters={iters}, 
        use_same_n_params={use_same_n_params},
        total_ntf_params={ntf_params},
        ntf_rank={ntf_rank},
        total_snmf_params={snmf_params},
        snmf_rank={snmf_rank},
        total_sntf_params={sntf_params},
        sntf_rank={topics_rank}
        """

        f.write(out)

    out = (
        [w_ntf, h_ntf],
        losses_ntf,
        [v_sntf, w_sntf, h_sntf],
        losses_sntf,
        [v_n, w_n, h_n],
        losses_snmf,
    )
    
    # save results
    with open(f"{path}/params.pickle", "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return out


def reconstruct_ntf(w, h):
    """Returns the NTF reconstruction of data tensor from w, h parameters."""
    return batch_tensor_product([w, *h]).sum(0)

def reconstruct_snmf(V, W, H):
    """Returns the SNMF reconstruction of data from V, W, H parameters."""
    return [np.dot(W[s], H) + np.outer(np.ones(len(W[s])), V[s]) for s in range(len(V))]


def plot_losses(
        ntf_losses: list = None, 
        sntf_losses: list = None, 
        snmf_losses: list = None, 
        path: str = "", 
        pip: bool = False, 
        **kwargs
) -> None:
    """Plots comparison of loss values over iterations
    
    Args:
        ntf_losses: List of classical ntf losses for each iteration. Defaults to None.
        sntf_losses: List of stratified ntf losses for each iteration. Defaults to None.
        snmf_losses: List of stratified nmf losses for each iteration. Defaults to None.
        path: File path to save figure. Defaults to current working directory.
        pip: Whether to include picture-in-picture plot starting from first iteration. Defaults to False.
        
    Returns: 
        None
    """
    xs = np.arange(len(sntf_losses))
    print(len(xs))
    markevery = list(range(0, len(sntf_losses), len(sntf_losses) // 5)) + [len(sntf_losses) - 1]

    fig, main_ax = plt.subplots(figsize=(5, 4), dpi=300)
    
    # adds picture in picture axis
    if pip:
        pip_ax = fig.add_axes([0.48, 0.45, 0.4, 0.4])

    # for each provided loss list, plot to main and pip axis

    if ntf_losses is not None:
        main_ax.plot(xs,
            ntf_losses,
            label="Classical NTF",
            color="tab:blue",
            marker="o",
            markevery=markevery,
        )
        print(f"Classical NTF final loss: {ntf_losses[-1]}")
        if pip:
            pip_ax.plot(xs[1:],
                ntf_losses[1:],
                label="Classical NTF",
                color="tab:blue",
                marker="o",
            )
            
            
    if snmf_losses is not None:
        main_ax.plot(xs,
            snmf_losses,
            label="Stratified-NMF",
            color="tab:orange",
            linestyle="dotted",
            marker="+",
            markevery=markevery,
        )
        print(f"Stratified NMF final loss: {snmf_losses[-1]}")
        if pip:
            pip_ax.plot(xs[1:],
                snmf_losses[1:],
                label="Stratified-NMF",
                color="tab:orange",
                linestyle="dotted",
                marker="+",
            )
            
    if sntf_losses is not None:
        main_ax.plot(xs,
            sntf_losses,
            label="Stratified-NTF",
            color="tab:green",
            linestyle="dashed",
            marker="*",
            markevery=markevery,
        )
        print(f"Stratified NTF final loss: {sntf_losses[-1]}")
        if pip:
            pip_ax.plot(xs[1:],
                sntf_losses[1:],
                label="Stratified-NTF",
                color="tab:green",
                linestyle="dashed",
                marker="*",
            )
            
    #main_ax.legend()
    main_ax.set_xlabel("Iterations")
    main_ax.set_ylabel("Loss")
    #plt.ylim(top=sntf_losses[1]*1.1)
    main_ax.semilogy()
    
    if pip:
        pip_ax.legend()
        pip_ax.set_xticks(xs[1:])
        pip_ax.locator_params(axis='y', nbins=10) 
    else:
        main_ax.legend()
    
    fig.savefig(f"{path}/loss_plot.png")
    fig.show()


def plot_reconstruct(
    A: np.ndarray, 
    ntf_params: list,
    sntf_params: list, 
    snmf_params: list, 
    s_ind: int, 
    o_ind: int, 
    path: str = "",
    suffix: str="",
    **kwargs
) -> plt.Figure:
    """Plots reconstructions for mode-3 image data.
    
    Args:
        A: Ground truth data tensor
        ntf_params: List of learned classical ntf parameters
        sntf_params: List of learned stratified ntf parameters
        snmf_params: List of learned stratified nmf parameters
        s_ind: Strata index to display
        o_ind: Observation index in stratum to display
        path: File path to save figure. Defaults to current working directory.
        suffix: String to be added to end of file name. Defaults to "".
    Returns:
        Figure object
    """

    # determines how many loss lists were provided
    n_provided = len([x for x in [ntf_params, sntf_params, snmf_params] if np.any(x[0])])

    n_strata = len(A)
    num_samples = [len(AA) for AA in A]
    s_inds = [0] + list(np.cumsum(num_samples))

    # plot ground truth image
    plt.figure(figsize=(5, 5))
    plt.imshow(A[s_ind][o_ind], **kwargs)
    plt.axis("off")
    plt.savefig(f"{path}/ground_truth_{s_ind}_{o_ind}{suffix}.png")
    plt.clf()

    fig, ax = plt.subplots(1, n_provided + 1, figsize=(15, 5))
    plt.axis("off")

    ax[0].imshow(A[s_ind][o_ind], **kwargs)
    ax[0].set_title("Ground truth")
    ax[0].axis("off")


    # for each provided method, plot the corresponding image reconstruction
    i = 1
    if np.any(ntf_params[0]):

        B_ntf = reconstruct_ntf(*ntf_params)
        B_ntf = [B_ntf[s_inds[i] : s_inds[i + 1]] for i in range(n_strata)]
        print(
            [
                np.round(np.sum((A[s_ind][i] - B_ntf[s_ind][i]) ** 2), 2)
                for i in range(10)
            ]
        )

        plt.figure(figsize=(5, 5))
        plt.imshow(B_ntf[s_ind][o_ind], **kwargs)
        plt.axis("off")
        plt.savefig(f"{path}/classical_ntf_{s_ind}_{o_ind}{suffix}.png")

        ax[i].imshow(B_ntf[s_ind][o_ind], **kwargs)
        ax[i].set_title("Classical NTF")
        ax[i].axis("off")

        i += 1

    if np.any(snmf_params[0]):
        B_snmf = reconstruct_snmf(*snmf_params)
        print(
            [
                np.round(
                    np.sum(
                        (A[s_ind][i] - B_snmf[s_ind][i].reshape(A[s_ind][i].shape)) ** 2
                    ),
                    2,
                )
                for i in range(10)
            ]
        )

        B_out = B_snmf[s_ind][o_ind].reshape(A[s_ind][o_ind].shape)

        plt.figure(figsize=(5, 5))
        plt.imshow(B_out, **kwargs)
        plt.axis("off")
        plt.savefig(f"{path}/stratified_nmf_{s_ind}_{o_ind}{suffix}.png")

        # plt.clf()

        ax[i].imshow(B_out, **kwargs)
        ax[i].set_title("Stratified NMF")
        ax[i].axis("off")
        i += 1

    if np.any(sntf_params[0]):
        B_sntf = compute_Bs(num_samples[s_ind], *sntf_params, s_ind)
        print([np.round(np.sum((A[s_ind][i] - B_sntf[i]) ** 2), 2) for i in range(10)])

        plt.figure(figsize=(5, 5))
        plt.imshow(B_sntf[o_ind], **kwargs)
        plt.axis("off")
        plt.savefig(f"{path}/stratified_ntf_{s_ind}_{o_ind}{suffix}.png")

        # plt.clf()

        ax[i].imshow(B_sntf[o_ind], **kwargs)
        ax[i].set_title("Stratified NTF")
        ax[i].axis("off")
        i += 1

    fig.savefig(f"{path}/reconstuction_comparison_{s_ind}_{o_ind}{suffix}.png")
    plt.axis("off")
    plt.show()
    return fig

def plot_stratified_nmf_global_topics(
        h_snmf: np.ndarray, 
        data_dims: tuple,
        path: str="",
        **kwargs
) -> None:
    """Plot Stratified NMF global topics for mode-3 image data."""
    for i, topic in enumerate(h_snmf):
        plt.figure(figsize=(5, 5), dpi=300)
        plt.imshow(topic.reshape(*data_dims), **kwargs)
        plt.axis("off")
        plt.savefig(f"{path}/stratified_nmf_topic_{i}.png")

def plot_strata_features(
        params: Iterable, 
        data_dims: tuple,
        path: str="",
        suffix: str="",
        **kwargs
) -> None:
    """Plots strata features for mode-3 image data from stratified ntf or nmf parameters."""
    
    # if stratified nmf parameters provided
    if isinstance(params[0], np.ndarray):
        for i, vv in enumerate(params[0].reshape(params[0].shape[0], *data_dims)):
            plt.figure(figsize=(5, 5), dpi=300)
            plt.imshow(vv, **kwargs)
            plt.axis("off")
            plt.savefig(f"{path}/stratified_nmf_strata_features_{i}{suffix}.png")
            plt.show()
            plt.close()

    # if stratified ntf parameters provided
    else:
        for i, vv in enumerate(params[0]):

            plt.figure(figsize=(5, 5), dpi=300)
            plt.imshow(batch_tensor_product(vv).sum(0), **kwargs)
            plt.axis("off")
            plt.savefig(f"{path}/stratified_ntf_strata_features_{i}{suffix}.png")
            plt.show()
            plt.close()
