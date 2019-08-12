# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from . import Simulation
from .AppWLC.Code import WLC

from scipy.interpolate import interp1d

class SegmentInfo(object):
    def __init__(self,list_of_segments,dZ_final):
        self.list_of_segments = list_of_segments
        self.dZ_final = dZ_final

class SimulatedSegmentFEC(object):
    def __init__(self,kw_simulation,force_function,kw_force):
        self.kw_simulation = kw_simulation
        self.force_function = force_function
        self.kw_force = kw_force
    @property
    def _interp_F(self):
        _interp_F = self.force_function(**self.kw_force)
        return _interp_F
    def _eq_potential(self,*args,**kwargs):
        return get_dV_dq(*args, f=equilibration_potential,**kwargs)

    def _forced_potential(self,*args,**kwargs):
        return get_dV_dq(*args, interp_F=self._interp_F,f=fec_potential,**kwargs)

def wlc_interpolator(Lp=0.3e-9,L0=200e-9,K0=100e-12,F_max=200e-12):
    F = np.linspace(0.1e-12,F_max,num=1000,endpoint=True)
    X = WLC.ExtensionPerForceOdjik(kbT=4.1e-21,Lp=Lp,L0=L0,K0=K0,
                                   F=F)
    interp_F = interp1d(x=X, y=F, kind='linear',fill_value=(0,F[-1]),
                        bounds_error=False)
    return interp_F


def fec_potential(q_n,z_n,k_L,x_i,k,interp_F):
    landscape = Simulation.dV_dq_i(q_n, z_n, 0, x_i, k)
    # also add in the WLC element
    force_WLC = interp_F(q_n)
    return landscape + force_WLC

def equilibration_potential(q_n,z_n,k_L,x_i,k):
    return Simulation.dV_dq_i(q_n, z_n, 0, 0, k)

def get_dV_dq(barrier_x,f, **kw):
    """
    see get_ks, except returns dV_dq(q,z)
    """
    n = len(barrier_x)
    arr = [Simulation._build_lambda(f, x_i=barrier_x[i], **kw)
           for i in range(n)]
    return arr


def _splice_f(a,b,f):
    f_a = f(a)
    f_b = f(b)
    combination = list(f_a) + list(f_b)
    return np.array(combination)

def _splice(a,b):
    t = _splice_f(a,b, lambda tmp: tmp.Time)
    x = _splice_f(a,b, lambda tmp: tmp.Extension)
    z = _splice_f(a,b, lambda tmp: tmp.ZSnsr)
    f = _splice_f(a,b, lambda tmp: tmp.Force)
    states =  _splice_f(a,b, lambda tmp: tmp.States)
    p = a.p
    return Simulation.SimpleFEC(t=t,x=x,z=z,f=f,p=p,states=states)

def _simulate_single_segment(segment,**kwargs):
    f_dV = segment._forced_potential
    to_ret =  Simulation.HummerSimpleFEC(f_dV=f_dV,**kwargs)
    return to_ret

def read_state(initial):
    t, f = initial.Time, initial.Force
    dt = t[1] - t[0]
    sim = Simulation.simulation_state(state=(initial.States[-1]+1) % 2,
                                      q_n=initial.Extension[-1],
                                      F_n=initial.Force[-1],
                                      k_n=0,
                                      dV_n=0,
                                      t=t[-1]+dt,
                                      i=initial.Force.size + 1)
    return sim

def _update_kw(initial,dZ,**kw):
    to_ret = dict(**kw)
    # move the zf function, so we can pkl
    z0 = initial.ZSnsr[-1]
    zf = z0 + dZ
    to_ret['z_0'] = z0
    to_ret['z_f'] = zf
    return to_ret

def _simulate_single_fec(list_of_segments,dZ_final):
    # get the initial segment
    assert len(list_of_segments) > 0
    kwargs = list_of_segments[0].kw_simulation
    initial = _simulate_single_segment(list_of_segments[0],**kwargs)
    for segment in list_of_segments[1:]:
        state_current = read_state(initial)
        # get the new keywords, update the z stuff...
        kwargs = segment.kw_simulation
        new = _simulate_single_segment(segment,state_current=state_current,
                                       skip_equil=True,**kwargs)
        initial = _splice(initial,new)
    N = len(list_of_segments)
    # add in a final segment at zero force.
    kwargs = _update_kw(initial,dZ=dZ_final,**kwargs)
    kwargs['f_dV'] = segment._eq_potential
    sim = read_state(initial)
    post = _generate_post_rupture(state_current=sim, skip_equil=True,
                                  **kwargs)
    # splice the two together
    to_ret = _splice(initial, post)
    to_ret._segment_info =  SegmentInfo(list_of_segments,dZ_final)
    return to_ret


def _generate_post_rupture(**kw):
    fecs = Simulation.HummerSimpleFEC(**kw)
    return fecs
