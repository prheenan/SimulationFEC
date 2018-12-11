# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import copy

from . import Simulation
from .AppWLC.Code import WLC
from scipy.interpolate import interp1d

class SimulateObject(object):
    def __init__(self,kw_simulate=dict(),kw_1_stretch=dict(),kw_2_stretch=None,
                 ignore_stretching=False):
        """
        :param kw_simulate: passed to  Simulation.HummerSimpleFEC
        :param kw_1_stretch: keywords to wlc_interpolator
        :param kw_2_stretch: keywords to wlc_interpolator
        """
        self.kw_sim = kw_simulate
        self.kw_1_stretch = kw_1_stretch
        if kw_2_stretch is None:
            kw_2_stretch = kw_1_stretch
        self.kw_2_stretch = kw_2_stretch
        # make the force interpolators
        self.interp_1 = wlc_interpolator(**self.kw_1_stretch)
        self.interp_2 = wlc_interpolator(**self.kw_2_stretch)
        self.ignore_stretching = ignore_stretching
    def dV_dq_n(self,interp,q_n,*args,**kwargs):
        """
        :param interp: interpolator used for getting the added force
        :param q_n:  the current extension
        :param args: the current argument
        :param kwargs: the current kwargs
        :return: the potential at the given extension
        """
        if self.ignore_stretching:
            to_add = 0
        else:
            to_add = interp(q_n)
        return to_add + Simulation.dV_dq_i(q_n,*args,**kwargs)
    @property
    def f_dV_dq_1(self):
        return lambda *args, **k: self.dV_dq_n(self.interp_1,*args,**k)
    @property
    def f_dV_dq_2(self):
        return lambda *args, **k: self.dV_dq_n(self.interp_2,*args,**k)
    def build_dV_dq(self):
        f_to_use = [ self.f_dV_dq_1, self.f_dV_dq_2]
        to_ret = lambda *args, **kw : Simulation.get_dV_dq(*args,f_to_use=f_to_use,**kw)
        return to_ret
    def __call__(self):
        return Simulation.HummerSimpleFEC(f_dV=self.build_dV_dq(),
                                          **self.kw_sim)


def wlc_interpolator(Lp=0.3e-9,L0=200e-9,K0=100e-12,F_max=200e-12,kbT=4.1e-21,
                     n_interp=2000):
    F = np.linspace(0.1e-12,F_max,num=n_interp,endpoint=True)
    X = WLC.ExtensionPerForceOdjik(kbT=kbT,Lp=Lp,L0=L0,K0=K0,
                                   F=F)
    interp_F = interp1d(x=X, y=F, kind='linear',fill_value=(0,F[-1]),
                        bounds_error=False)
    return interp_F

def fec_potential(q_n,z_n,k_L,x_i,k,interp_F):
    landscape = Simulation.dV_dq_i(q_n, z_n, k_L, x_i, k)
    # also add in the WLC element
    force_WLC = interp_F(q_n)
    return landscape + force_WLC

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

def _simulate_single_fec(list_of_segments):
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
    return initial


