# miscellaneous useful functions and classes
import numpy as np
import os


class Generic(object):
    """Class for generic object."""
    
    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            self.__dict__[k] = v


def c_tile(x, n):
    """Create tiled matrix where each of n cols is x."""
    return np.tile(x.flatten()[:, None], (1, n))


def r_tile(x, n):
    """Create tiled matrix where each of n rows is x."""
    return np.tile(x.flatten()[None, :], (n, 1))


def get_idx(z, z_0, dz, l):
    """
    Return closest valid integer index of continuous value.
    
    :param z: continuous value
    :param z_0: min continuous value (counter start point)
    :param dz: 
    """
    try:
        z[0]
    except:
        z = np.array([z]).astype(float)
        
    int_repr = np.round((z-z_0)/dz).astype(int)
    return np.clip(int_repr, 0, l-1)
