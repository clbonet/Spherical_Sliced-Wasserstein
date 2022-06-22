import torch
import numpy as np

from sklearn.datasets import make_circles



def rand_ring2d(batch_size):
    """ This function generates 2D samples from a hollowed-cirlce distribution in a 2-dimensional space.
        Args:
            batch_size (int): number of batch samples
        Return:
            torch.Tensor: tensor of size (batch_size, 2)

        ## https://github.com/eifuentes/swae-pytorch/blob/763f771c1d4860f71819af48d4f21a8a29a689d5/swae/distributions.py#L38
    """
    circles = make_circles(2 * batch_size, noise=.01)
    z = np.squeeze(circles[0][np.argwhere(circles[1] == 0), :])
    return torch.from_numpy(z).type(torch.FloatTensor)


def rand_circle2d(batch_size):
    """ This function generates 2D samples from a filled-circle distribution in a 2-dimensional space.
        Args:
            batch_size (int): number of batch samples
        Return:
            torch.Tensor: tensor of size (batch_size, 2)
        
        ## https://github.com/eifuentes/swae-pytorch/blob/763f771c1d4860f71819af48d4f21a8a29a689d5/swae/distributions.py#L38
    """
    r = np.random.uniform(size=(batch_size))
    theta = 2 * np.pi * np.random.uniform(size=(batch_size))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.array([x, y]).T
    return torch.from_numpy(z).type(torch.FloatTensor)