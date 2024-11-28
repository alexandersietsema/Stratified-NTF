# olivetti_faces.py
"""
This file contains code to run the Olivetti Faces experiment.
"""
from sklearn.datasets import fetch_olivetti_faces
from plotting import plot_losses, plot_reconstruct, plot_strata_features, run_experiment, make_directory
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data = fetch_olivetti_faces()
    Astack = data.data.reshape(400, 64, 64)
    
    # create new directory for experiment
    path = make_directory("faces")

    # stratify the data by individual
    A = [Astack[data.target == i] for i in range(40)]
    data_dims = A[0].shape[1:]
    
    # run methods
    ntf_params, losses_ntf, sntf_params, losses_sntf, snmf_params, losses_snmf =\
        run_experiment(A, iters=1000, strata_feature_rank = 15, topics_rank=40, path=path)

    kwargs = dict(vmin=0, vmax=1)
    # make convergence plots
    plot_losses(losses_ntf[:-1], losses_sntf, losses_snmf, path=path)
    
    # plot reconstructions for the first pair with different viewing angles
    plot_reconstruct(A, ntf_params, sntf_params, snmf_params, 0, 0, path=path,**kwargs)
    plot_reconstruct(A, ntf_params, sntf_params, snmf_params, 0, 3, path=path,**kwargs)
    
    # plot reconstructions for two other individuals
    plot_reconstruct(A, ntf_params, sntf_params, snmf_params, 1, 0, path=path,**kwargs)
    plot_reconstruct(A, ntf_params, sntf_params, snmf_params, 2, 0, path=path,**kwargs)
    
    # plot strata features for SNMF and SNTF
    plot_strata_features(sntf_params, data_dims, path=path, **kwargs)
    plot_strata_features(snmf_params, data_dims, path=path, **kwargs)
    plt.clf()
    