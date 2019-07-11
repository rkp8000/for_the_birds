"""
Classes/functions for a few biological spiking network models.
"""
from copy import deepcopy as copy
import numpy as np
from scipy.sparse import csc_matrix

from aux import Generic, c_tile, r_tile

cc = np.concatenate


# Binary spiking network
class BinarySTDPNtwk(object):
    """Network of binary spiking neurons."""
    
    def __init__(self, cxn, w_0, tht_s, t_tht, a_tht, n_max, t_stdp, d_w_s, w_min, w_max):
        self.cxn = cxn
        self.w_0 = w_0
        self.tht_s = tht_s
        self.t_tht = t_tht
        self.a_tht = a_tht
        self.n_max = n_max
        self.t_stdp = t_stdp
        self.d_w_s = d_w_s
        self.w_min = w_min
        self.w_max = w_max
        
        self.n = w_0.shape[0]
        
        self.t_stdp_ub = t_stdp[-1] + 1
        self.t_stdp_lb = t_stdp[0]
        
        self.d_w_s_p = d_w_s[t_stdp >= 0]
        self.d_w_s_m = d_w_s[t_stdp < 0]
    
    def run(self, i_ext, t_save_w=None, f_w_save=None, change_w=True):
        """
        Run network.
        
        Set 'change_w' to True to calc and apply stdp, to False to calc but not apply stdp,
        or to None to neither calc nor apply stdp.
        """
        cxn = self.cxn
        w_0 = self.w_0
        tht_s = self.tht_s
        t_tht = self.t_tht
        a_tht = self.a_tht
        n_max = self.n_max
        t_stdp = self.t_stdp
        d_w_s = self.d_w_s
        w_min = self.w_min
        w_max = self.w_max
        
        n = self.n
        
        t_stdp_ub = self.t_stdp_ub
        t_stdp_lb = self.t_stdp_lb
        
        d_w_s_p = self.d_w_s_p
        d_w_s_m = self.d_w_s_m
        
        n_t = len(i_ext)
        
        if t_save_w is None:
            t_save_w = [1, n_t-1]
        
        vs = np.zeros((n_t, n))
        thts = tht_s * np.ones((n_t, n))
        spks = np.zeros((n_t, n), dtype=bool)
        
        w = w_0.copy()
        d_w_s = np.zeros((n, n))
        
        f_ws = [] if f_w_save is not None else None
        
        ws = {}
        
        for t in range(1, n_t):

            # get inputs
            ## recurrent
            i_rcr = w.dot(spks[t-1, :])

            ## total
            v = i_ext[t] + i_rcr

            # thresholds
            tht = thts[t-1, :] - (1/t_tht)*(thts[t-1, :] - tht_s) + a_tht*spks[t-1, :]

            # mask n_max highest vs
            v_max = np.zeros(n, dtype=bool)
            v_max[np.argsort(v)[-n_max:]] = True

            # get spks
            spk = (v >= tht) & v_max  # spk if v above threshold and included in max vs

            # save dynamic vars
            vs[t, :] = v.copy()
            thts[t, :] = tht.copy()
            spks[t, :] = spk.copy()
            
            # calc w update
            if change_w is not None:
                # increase w where post spikes occurred
                mask_p = cxn & c_tile(spk, n)  # cxns w/ post spks

                if t < t_stdp_ub-1:
                    h_stdp = d_w_s_p[:t+1][::-1]  # pos-lobe STDP filter to conv w spks
                    d_w_s_p_ = r_tile(np.dot(h_stdp, spks[:t+1, :]), n)  # \Delta w^*_+
                else:
                    h_stdp = d_w_s_p[::-1]
                    d_w_s_p_ = r_tile(np.dot(h_stdp, spks[t-t_stdp_ub+1:t+1, :]), n)

                # decrease w where pre spikes occurred
                mask_m = cxn * r_tile(spk, n)  # cxns w/ pre spks

                if t < -(t_stdp_lb):
                    h_stdp = d_w_s_m[len(d_w_s_m)-t:]  # minus-lobe STDP filter to conv w spks
                    d_w_s_m_ = c_tile(np.dot(h_stdp, spks[:t, :]), n)  # \Delta w^*_-
                else:
                    h_stdp = d_w_s_m
                    d_w_s_m_ = c_tile(np.dot(h_stdp, spks[t+t_stdp_lb:t, :]), n)
                    
                d_w_s[mask_p] += d_w_s_p_[mask_p]
                d_w_s[mask_m] += d_w_s_m_[mask_m]

            # apply w update if desired
            if change_w:
                w[mask_p] += (w_max - w[mask_p]) * d_w_s_p_[mask_p]  # scale ∆w^*_+ by dist to w_max
                w[mask_m] += (w[mask_m] - w_min) * d_w_s_m_[mask_m]  # scale ∆w^*_- by dist from w_min
                
            # save weights

            ## [distribution]
            if f_w_save is not None:
                f_ws.append(f_w_save(w))

            ## [full matrix]
            if t in t_save_w:
                ws[t] = w.copy()

        return Generic(t=np.arange(n_t), vs=vs, spks=spks, thts=thts, d_w_s=d_w_s, f_ws=f_ws, ws=ws)


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
        
        if spks_u is not None:
            assert len(i_ext) == len(spks_u)
        
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
                    spks_w = w_r[syn].dot(spks[t_ctr-1, :])
                    ## upstream
                    if spks_u is not None:
                        if syn in w_u:
                            spks_w += w_u[syn].dot(spks_u[t_ctr-1, :])
                    
                    # update conductances from weighted spks
                    gs[syn][t_ctr, :] = g + (dt/t_s[syn])*(-gs[syn][t_ctr-1, :]) + spks_w
            
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
        
        return Generic(dt=dt, t=t, gs=gs, vs=vs, spks=spks, spks_t=spks_t, spks_c=spks_c, i_ext=i_ext, ntwk=self)


# Adaptive integrate-and-fire network (current-based)
class ALIFNtwkI(object):
    """
    Network of *adaptive* leaky integrate-and-fire neurons with *current-based* synapses.
    
    Based on Brette and Gerstner J, Neurophysiol 2005.
    (https://www.ncbi.nlm.nih.gov/pubmed/16014787)
    """
    
    def __init__(self, c_m, g_l, e_l, v_th, v_r, t_r, t_a, b_v, b_s, w_r, w_u, sparse=True):
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
        self.t_a = t_a
        self.b_v = b_v
        self.b_s = b_s
        
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
        t_a = self.t_a
        b_v = self.b_v
        b_s = self.b_s
        w_r = self.w_r
        w_u = self.w_u
        
        if spks_u is not None:
            assert len(i_ext) == len(spks_u)
        
        # make data storage arrays
        vs = np.nan * np.zeros((n_t, n))
        spks = np.zeros((n_t, n), dtype=bool)
        
        rp_ctr = np.zeros(n, dtype=int)  # refractory counter
        a = np.zeros(n)  # adaptation variable
        
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
                i_total += a  # adaptation
                
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
            
            # update adaptation variable
            a += (dt/t_a)*(-a + b_v*(vs[t_ctr, :] - e_l)) + b_s*spks[t_ctr]
        
        t = dt*np.arange(n_t, dtype=float)
        
        # convert spks to spk times and cell idxs (for easy access l8r)
        tmp = spks.nonzero()
        spks_t = dt * tmp[0]
        spks_c = tmp[1]
        
        return Generic(dt=dt, t=t, vs=vs, spks=spks, spks_t=spks_t, spks_c=spks_c, i_ext=i_ext, ntwk=self)
    
    
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
