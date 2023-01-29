# Spherical Sliced-Wasserstein

This repository contains the code and the experiments of the paper [Spherical Sliced-Wasserstein](https://arxiv.org/abs/2206.08780). We propose in this paper a new Sliced-Wasserstein type of discrepancy on the sphere. We illustrate its properties on several machine learning tasks such as density estimation, variational inference or hyperspherical auto-encoders. We also test it on a Self-supervised learning task.

## Abstract

Many variants of the Wasserstein distance have been introduced to reduce its original computational burden. In particular the Sliced-Wasserstein distance (SW), which leverages one-dimensional projections for which a closed-form solution of the Wasserstein distance is available, has received a lot of interest. Yet, it is restricted to data living in Euclidean spaces, while the Wasserstein distance has been studied and used recently on manifolds. We focus more specifically on the sphere, for which we define a novel SW discrepancy, which we call spherical Sliced- Wasserstein, making a first step towards defining SW discrepancies on manifolds. Our construction is notably based on closed-form solutions of the Wasserstein distance on the circle, together with a new spherical Radon transform. Along with efficient algorithms and the corresponding implementations, we illustrate its properties in several machine learning use cases where spherical representations of data are at stake: density estimation on the sphere, variational inference or hyperspherical auto-encoders.

## Citation

```
@inproceedings{bonet2023spherical,
    title={Spherical Sliced-Wasserstein},
    author={Clément Bonet and Paul Berg and Nicolas Courty and François Septier and Lucas Drumetz and Minh-Tan Pham},
    year={2023},
    booktitle={International Conference on Learning Representations},
}
```

## Experiments

- In the folder "Runtime - Evolutions", you can find some experiments on SSW.
- In the GradientFlows notebook, you can find experiments of Section 5.1 where we aim at learning a distribution on the sphere from which we have access to samples.
- In the Density Estimation folder, we report the code used for density estimation used in Section 5.1.
- In the "SWVI" folder, you can find the code of the variational inference experiment of Appendix X.5. In this experiment, we want to learn a distribution from which we known the density up to a constant.
- In the folder "SWAE", you can find the code to reproduce the experiments of Section 5.2 in which we compare several Wasserstein autoencoders with different divergence between the prior and the generator in the latent space. More precisely, we compared SSW with SW, MMD and the IMQ or RBF kernel, the Sinkhorn divergence and the (circular) generalized SW distance.


## Credits

- For the code of the 1D circular Wasserstein distance, inspirations were taken from the matlab code of the original paper (https://users.mccme.ru/ansobol/otarie/software.html) for the binary search, and from the [CircularOT](https://gitlab.gwdg.de/shundri/circularOT/-/tree/master/) repository for the level median formulation.
- The code of the exponential map normalizing flow was inspired by the [sdflows](https://github.com/katalinic/sdflows) repository as well as from the [rcpm](https://github.com/facebookresearch/rcpm) repository.
- The generalized SW distance code was taken from the [gsw](https://github.com/kimiandj/gsw) repository.
- For the von Mises-Fisher distributions, we used some code of the [von-Mises-Sampling](https://github.com/dlwhittenbury/von-Mises-Sampling) repository and of the [von-Mises-Fisher-Sampling](https://github.com/dlwhittenbury/von-Mises-Fisher-Sampling) repository. For the power spherical distribution, we used some code of the [power_spherical](https://github.com/nicola-decao/power_spherical) repository.
- For the FID implementation, we used some code of the [SINF](https://github.com/biweidai/SINF) repository.
- To load the earth data, we used some code of the [moser_flow](https://github.com/noamroze/moser_flow) repository.

