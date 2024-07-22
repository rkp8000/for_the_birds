import numpy as np
from scipy import stats


def get_cmn_p(a, p):
    """
    Circular mean from prob distr.
    
    a: angle vector of form np.arange(0, 2*np.pi, 2*np.pi/n)
    p: corresponding probabilities
    """
    return np.arctan2(p@np.sin(a), p@np.cos(a))


def get_c_spd(c_mn, t, t_start):
    """Estimate speed of uniform circular motion.
    c_mn in radians
    """
    c_mn_unw = np.unwrap(c_mn)
    
    slp = stats.linregress(t[t_start <= t], c_mn_unw[t_start <= t])[0]
    
    return slp
