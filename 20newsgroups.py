#20newsgroups.py
"""
This file contains code to run the 20 Newsgroups experiments.
"""

import numpy as np
import sklearn
import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from plotting import plot_losses, run_experiment, make_directory

def get_20newsgroups_data(n_docs=100):
    """Construct the stratified 20 newsgroups dataset."""
    data, x = sklearn.datasets.fetch_20newsgroups(
        remove=("headers", "footers", "quotes"), return_X_y=True, subset="train"
    )
    
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, max_df=0.95)
    A_full = vectorizer.fit_transform(data)

    A = []
    # Construct stratified data from each subtopic corresponding to each supertopic
    A.append(np.stack([A_full[x == i][:n_docs].toarray() for i in range(1, 6)]))
    A.append(np.stack([A_full[x == i][:n_docs].toarray() for i in range(7, 11)]))
    A.append(np.stack([A_full[x == i][:n_docs].toarray() for i in range(11, 15)]))
    A.append(np.stack([A_full[x == i][:n_docs].toarray() for i in range(16, 19)]))
    A.append(np.stack([A_full[x == i][:n_docs].toarray() for i in [0, 15, 19]]))
    
    return A, vectorizer

def get_top_words(x, features, n):
    """Get the top n highest correspondence words for each topic."""
    return [features[i] for i in np.argsort(-x)[:n]]

if __name__ == "__main__":

    path = make_directory("20newsgroups")
    A, vectorizer = get_20newsgroups_data()

    # run methods
    ntf_params, losses_ntf, sntf_params, losses_sntf, snmf_params, losses_snmf = (
        run_experiment(
            A,
            iters=11,
            strata_feature_rank=3,
            topics_rank=5,
            path=path,
            do_stratified_nmf=False,
            do_classical_ntf=False,
        )
    )
    
    # Plot the loss 
    plot_losses(losses_ntf, losses_sntf, losses_snmf, path=path)
    v_sntf, w_sntf, h_sntf = sntf_params
    
    # for each stratum, print the 3 highest correspondence words
    for i in range(5):
        for j in range(3):
            print(get_top_words(v_sntf[i][-1][j], vectorizer.get_feature_names_out(), 3))
        print()
