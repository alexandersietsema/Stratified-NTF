# mnist.py
"""
This implements stratified NTF on the noisy watermarked MNIST dataset.
"""
import matplotlib.pyplot as plt
import numpy as np
from data import get_mnist_watermarked
from data import load_numeric_watermarks
from stratified_ntf import stratified_ntf
from plotting import plot_reconstruct, plot_strata_features, make_directory


if __name__ == "__main__":

    # Run stratified NTF on MNIST
    n_images = 100
    
    noise_level = 0.15
    
    watermarks = load_numeric_watermarks()[[0, 4]]
    data = get_mnist_watermarked({"12": [1, 2], "23": [2, 3]}, n_images, watermarks)

    # Create a copy of the tensor normalized to [0,1]
    noisy_data = [np.copy(d.astype(np.float64) / 255) for d in data]
    
    def salt_and_pepper(im: np.ndarray, p: float = 0.15):
        """Adds salt-and-pepper noise to a [0,1] array."""
        out = im + np.random.choice([0.,1., -1], p=[1-2*p, p, p], size=im.shape)
        return np.clip(out, 0., 1.)
    
    A = [salt_and_pepper(AA, noise_level) for AA in noisy_data]
    
    # Run stratified NTF

    strata = 2
    num_samples = [2*n_images]*2
    iters = 100
    strata_feature_rank = [100]*strata
    topics_rank = 100
    fixed_dimensions = (28,28)
    
    path = make_directory('noisy_mnist')
    
    # use same starting point for each regularization level
    V = [
        [np.random.random((strata_feature_rank[i], d)) for d in fixed_dimensions]
        for i in range(strata)
    ]
    W = [np.random.random((topics_rank, num_samples[i])) for i in range(strata)]

    H = [np.random.random((topics_rank, d)) for d in fixed_dimensions]
    
    regs = [0.0, 5.0, 10.0]
    
    # log experiment parameter settings in txt file
    with open(f"{path}/params.txt", "w") as f:

        out = f"""
        strata_feature_rank={strata_feature_rank},
        topics_rank={topics_rank},
        iters={iters}, 
        regs={regs},
        n_images={n_images}
        noise_level={noise_level}
        """

        f.write(out)
    
    # run method for each regularization level
    for reg in regs:

        v, w, h, loss_array = stratified_ntf(
            A,
            strata_feature_rank,
            topics_rank,
            iters,
            v_scaling=2,
            starting=(V.copy(),W.copy(),H.copy()),
            reg_type="tv",
            reg = reg
        )
        
        kwargs = dict(vmin=0, vmax=1)
        
        # show reconstructions for first and second strata
        plot_reconstruct(A, [None], (v,w,h), [None], 0, 0, path=path, suffix=f"_{reg}", **kwargs)
        plot_reconstruct(A, [None], (v,w,h), [None], 0, 1, path=path, suffix=f"_{reg}", **kwargs)
        plot_reconstruct(A, [None], (v,w,h), [None], 0, 2, path=path, suffix=f"_{reg}", **kwargs)
        plot_reconstruct(A, [None], (v,w,h), [None], 1, 0, path=path, suffix=f"_{reg}", **kwargs)
        plot_reconstruct(A, [None], (v,w,h), [None], 1, 1, path=path, suffix=f"_{reg}", **kwargs)
        plot_reconstruct(A, [None], (v,w,h), [None], 1, 2, path=path, suffix=f"_{reg}", **kwargs)
        
        plot_strata_features((v,w,h), data_dims = (28,28), path=path, suffix=f"_{reg}")
        
        for i, hh in enumerate(h):
            fig, ax = plt.subplots(1,1,figsize=(5,5), dpi=300)
            ax.imshow(hh)
            ax.axis('off')
            plt.show()
            fig.savefig(f'{path}/topic_vectors_{i}_{reg}.png')
            plt.close()