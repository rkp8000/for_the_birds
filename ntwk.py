"""
Classes/functions for a few biological spiking network models.
"""
from copy import deepcopy as copy
import numpy as np
from scipy.sparse import csc_matrix

from aux import Generic, c_tile, r_tile

cc = np.concatenate


# Current-based LIF network
class LIFNtwkI(object):
    """Network of leaky integrate-and-fire neurons with *current-based* synapses."""
    
    def __init__(self, c_m, g_l, e_l, v_th, v_r, t_r, w_r, w_u, sparse=True):
        # ntwk size
        n = w_r.shape[0]
        
        # process inputs
        if type(t_r) in [float, int]:
            t_r = t_r * np.ones(n)
            
        if type(v_r) in [float, int]:
            v_r = v_r * np.ones(n)
            
        self.n = n
        self.c_m = c_m
        self.g_l = g_l
        self.t_m = c_m / g_l
        self.e_l = e_l
        self.v_th = v_th
        self.v_r = v_r
        self.t_r = t_r
        
        if sparse:  # sparsify connectivity if desired
            self.w_r = csc_matrix(w_r)
            self.w_u = csc_matrix(w_u) if w_u is not None else w_u
        else:
            self.w_r = w_r
            self.w_u = w_u
        
    def run(self, dt, clamp, i_ext, spks_u=None):
        """
        Run simulation.
        
        :param dt: integration timestep (s)
        :param clamp: dict of times to clamp certain variables (e.g. to initialize)
        :param i_ext: external current inputs (either 1D or 2D array, length = num timesteps for smln)
        :param spks_up: upstream inputs
        """
        n = self.n
        n_t = len(i_ext)
        c_m = self.c_m
        g_l = self.g_l
        e_l = self.e_l
        v_th = self.v_th
        v_r = self.v_r
        t_r = self.t_r
        t_r_int = np.round(t_r/dt).astype(int)
        w_r = self.w_r
        w_u = self.w_u
        
        if spks_u is not None:
            assert len(i_ext) == len(spks_u)
        
        # make data storage arrays
        vs = np.nan * np.zeros((n_t, n))
        spks = np.zeros((n_t, n), dtype=bool)
        
        rp_ctr = np.zeros(n, dtype=int)
        
        # convert float times in clamp dict to time idxs
        ## convert to list of tuples sorted by time
        tmp_v = sorted(list(clamp.v.items()), key=lambda x: x[0])
        tmp_spk = sorted(list(clamp.spk.items()), key=lambda x: x[0])
        clamp = Generic(
            v={int(round(t_/dt)): f_v for t_, f_v in tmp_v},
            spk={int(round(t_/dt)): f_spk for t_, f_spk in tmp_spk})
        
        # loop over timesteps
        for t_ctr in range(len(i_ext)):
            
            # update voltages
            if t_ctr in clamp.v:  # check for clamped voltages
                vs[t_ctr, :] = clamp.v[t_ctr]
            else:  # update as per diff eq
                v = vs[t_ctr-1, :]
                
                # get total current input
                i_total = -g_l*(v - e_l)  # leak
                
                if t_ctr >= 1:  # synaptic
                    
                    if spks_u is not None:  # upstream
                        i_total += w_u.dot(spks_u[t_ctr-1, :])
                    i_total += w_r.dot(spks[t_ctr-1, :])  # recurrent
                
                i_total += i_ext[t_ctr]  # external
                
                # update v
                vs[t_ctr, :] = v + (dt/c_m)*i_total
                
                # clamp v for cells still in refrac period
                vs[t_ctr, rp_ctr > 0] = self.v_r[rp_ctr > 0]
            
            # update spks
            if t_ctr in clamp.spk:  # check for clamped spikes
                spks[t_ctr, :] = clamp.spk[t_ctr]
            else:  # check for threshold crossings
                spks[t_ctr, :] = vs[t_ctr, :] >= self.v_th
                
            # reset v and update refrac periods for nrns that spiked
            vs[t_ctr, spks[t_ctr]] = self.v_r[spks[t_ctr]]
            rp_ctr[spks[t_ctr]] = t_r_int[spks[t_ctr]] + 1
            
            # decrement refrac periods
            rp_ctr[rp_ctr > 0] -= 1
            
            # update aux variables and weights
            # NOT IMPLEMENTED YET
        
        t = dt*np.arange(n_t, dtype=float)
        
        # convert spks to spk times and cell idxs (for easy access l8r)
        tmp = spks.nonzero()
        spks_t = dt * tmp[0]
        spks_c = tmp[1]
        
        return Generic(dt=dt, t=t, vs=vs, spks=spks, spks_t=spks_t, spks_c=spks_c, i_ext=i_ext, ntwk=self)

    
# Conductance-based LIF network
class LIFNtwkG(object):
    """Network of leaky integrate-and-fire neurons with *conductance-based* synapses."""
    
    def __init__(self, c_m, g_l, e_l, v_th, v_r, t_r, e_s, t_s, w_r, w_u, sparse=True):
        # ntwk size
        n = next(iter(w_r.values())).shape[0]
        
        # process inputs
        if type(t_r) in [float, int]:
            t_r = t_r * np.ones(n)
            
        if type(v_r) in [float, int]:
            v_r = v_r * np.ones(n)
            
        self.n = n
        self.c_m = c_m
        self.g_l = g_l
        self.t_m = c_m / g_l
        self.e_l = e_l
        self.v_th = v_th
        self.v_r = v_r
        self.t_r = t_r
        
        self.e_s = e_s
        self.t_s = t_s
        
        if sparse:  # sparsify connectivity if desired
            self.w_r = {k: csc_matrix(w_r_) for k, w_r_ in w_r.items()}
            self.w_u = {k: csc_matrix(w_u_) for k, w_u_ in w_u.items()} if w_u is not None else w_u
        else:
            self.w_r = w_r
            self.w_u = w_u

        self.syns = list(self.e_s.keys())
        
    def run(self, dt, clamp, i_ext, spks_u=None):
        """
        Run simulation.
        
        :param dt: integration timestep (s)
        :param clamp: dict of times to clamp certain variables (e.g. to initialize)
        :param i_ext: external current inputs (either 1D or 2D array, length = num timesteps for smln)
        :param spks_up: upstream inputs
        """
        n = self.n
        n_t = len(i_ext)
        syns = self.syns
        c_m = self.c_m
        g_l = self.g_l
        e_l = self.e_l
        v_th = self.v_th
        v_r = self.v_r
        t_r = self.t_r
        t_r_int = np.round(t_r/dt).astype(int)
        e_s = self.e_s
        t_s = self.t_s
        w_r = self.w_r
        w_u = self.w_u
        
        # make data storage arrays
        gs = {syn: np.nan * np.zeros((n_t, n)) for syn in syns}
        vs = np.nan * np.zeros((n_t, n))
        spks = np.zeros((n_t, n), dtype=bool)
        
        rp_ctr = np.zeros(n, dtype=int)
        
        # convert float times in clamp dict to time idxs
        ## convert to list of tuples sorted by time
        tmp_v = sorted(list(clamp.v.items()), key=lambda x: x[0])
        tmp_spk = sorted(list(clamp.spk.items()), key=lambda x: x[0])
        clamp = Generic(
            v={int(round(t_/dt)): f_v for t_, f_v in tmp_v},
            spk={int(round(t_/dt)): f_spk for t_, f_spk in tmp_spk})
        
        # loop over timesteps
        for t_ctr in range(len(i_ext)):
            
            # update conductances
            for syn in syns:
                if t_ctr == 0:
                    gs[syn][t_ctr, :] = 0
                else:
                    g = gs[syn][t_ctr-1, :]
                    # get weighted spike inputs
                    ## recurrent
                    inp = w_r[syn].dot(spks[t_ctr-1, :])
                    ## upstream
                    if spks_u is not None:
                        if syn in w_u:
                            inp += w_u[syn].dot(spks_u[t_ctr-1, :])
                    
                    # update conductances from weighted spks
                    gs[syn][t_ctr, :] = g + (dt/t_s[syn])*(-gs[syn][t_ctr-1, :]) + inp
            
            # update voltages
            if t_ctr in clamp.v:  # check for clamped voltages
                vs[t_ctr, :] = clamp.v[t_ctr]
            else:  # update as per diff eq
                v = vs[t_ctr-1, :]
                # get total current input
                i_total = -g_l*(v - e_l)  # leak
                i_total += np.sum([-gs[syn][t_ctr, :]*(v - e_s[syn]) for syn in syns], axis=0)  # synaptic
                i_total += i_ext[t_ctr]  # external
                
                # update v
                vs[t_ctr, :] = v + (dt/c_m)*i_total
                
                # clamp v for cells still in refrac period
                vs[t_ctr, rp_ctr > 0] = v_r[rp_ctr > 0]
            
            # update spks
            if t_ctr in clamp.spk:  # check for clamped spikes
                spks[t_ctr, :] = clamp.spk[t_ctr]
            else:  # check for threshold crossings
                spks[t_ctr, :] = vs[t_ctr, :] >= v_th
                
            # reset v and update refrac periods for nrns that spiked
            vs[t_ctr, spks[t_ctr, :]] = v_r[spks[t_ctr, :]]
            rp_ctr[spks[t_ctr, :]] = t_r_int[spks[t_ctr, :]] + 1
            
            # decrement refrac periods
            rp_ctr[rp_ctr > 0] -= 1
            
        t = dt*np.arange(n_t, dtype=float)
        
        # convert spks to spk times and cell idxs (for easy access l8r)
        tmp = spks.nonzero()
        spks_t = dt * tmp[0]
        spks_c = tmp[1]
        
        return Generic(dt=dt, t=t, gs=gs, vs=vs, spks=spks, spks_t=spks_t, spks_c=spks_c, i_ext=i_ext, ntwk=self)

    
# Helper functions
def join_w(targs, srcs, ws):
    """
    Combine multiple weight matrices specific to pairs of populations
    into a single, full set of weight matrices (one per synapse type).
    
    :param targs: dict of boolean masks indicating targ cell classes
    :param srcs: dict of boolean masks indicating source cell classes
    :param ws: dict of inter-population weight matrices, e.g.:
        ws = {
            'E': {
                ('E', 'E'): np.array([[...]]),
                ('I', 'E'): np.array([[...]]),
            },
            'I': {
                ('E', 'I'): np.array([[...]]),
                ('I', 'I'): np.array([[...]]),
            }
        }
        note: keys given as (targ, src)
    
    :return: ws_full, a dict of full ws, one per synapse
    """
    # convert targs/srcs to dicts if given as arrays
    if not isinstance(targs, dict):
        targs_ = copy(targs)
        targs = {
            cell_type: targs_ == cell_type for cell_type in set(targs_)
        }
    if not isinstance(srcs, dict):
        srcs_ = copy(srcs)
        srcs = {
            cell_type: srcs_ == cell_type for cell_type in set(srcs_)
        }
        
    # make sure all targ/src masks have same shape
    targ_shapes = [mask.shape for mask in targs.values()]
    src_shapes = [mask.shape for mask in srcs.values()]
    
    if len(set(targ_shapes)) > 1:
        raise Exception('All targ masks must have same shape.')
        
    if len(set(src_shapes)) > 1:
        raise Exception('All targ masks must have same shape.')
        
    n_targ = targ_shapes[0][0]
    n_src = src_shapes[0][0]
    
    # make sure weight matrix dimensions match sizes
    # of targ/src classes
    for syn, ws_ in ws.items():
        for (targ, src), w_ in ws_.items():
            if not w_.shape == (targs[targ].sum(), srcs[src].sum()):
                raise Exception(
                    'Weight matrix for {}: ({}, {}) does not match '
                    'dimensionality specified by targ/src masks.')
        
    # loop through synapse types
    dtype = list(list(ws.values())[0].values())[0].dtype
    ws_full = {}
    
    for syn, ws_ in ws.items():
        
        w = np.zeros((n_targ, n_src), dtype=dtype)
        
        # loop through population pairs
        for (targ, src), w_ in ws_.items():
            
            # get mask of all cxns from src to targ
            mask = np.outer(targs[targ], srcs[src])
            
            assert mask.sum() == w_.size
            
            w[mask] = w_.flatten()
            
        ws_full[syn] = w
        
    return ws_full
