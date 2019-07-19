# DoG-CNNs
Difference of Gaussian (DoG) filters as Convolutional Filters.

I just wanted to try out parametrizing the convolutional kernels as differences of Gaussians and I was suprised to find to find nothing online.

The parametrization as DoG leads to 11 Parameters per Kernel (3 for Covariance, 2 for mean = 5 per Gaussian plus 1 for the ratio between the two Gaussians per kernel), so ideally faster convergence.
These Kernels are size invariant, it is possible to adapt them to the resolution of the image.
My assumption was, that this way of parametrizing exploits the fact that neighbouring weights in a filter are highly correlated, while not losing too much expressiveness. This could possibly also improves the initialization of the weights.

Apart from possibly losing expressiveness, the biggest drawback is, that the filter-weights have to be calculated after each update, leading to slower training (in my implementation a factor of about 5).

So anyway, here is my implementation in Pytorch.
