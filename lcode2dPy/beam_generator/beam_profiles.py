import scipy.stats as stats
import numpy as np
import re
import os
from numpy import sqrt, pi
from copy import copy

########## Particle distributions #############
def RejectSamplDistr(distr, xmax, vmin, vmax):  # Doesn't work for xi distributions
    def generate(pdf, xmax, vmin, vmax, size):
        x = np.random.uniform(vmin, vmax, size=size)
        y = np.random.uniform(0, pdf(xmax), size=size)
        vals = pdf(x)
        selected = x[vals >= y]
        return selected

    N0 = 10000
    ratio = generate(distr, xmax, vmin, vmax, N0).size / N0

    def distr_maker(N):
        vals = generate(distr, xmax, vmin, vmax, int(N / ratio) * 2)
        return vals[:int(N)]
    distr_maker.name = 'arbitrary'
    distr_maker.xmax = xmax
    distr_maker.f = distr
    return distr_maker


def SmoothDistr(distr, vmin, vmax, *args, xi_max=None, amp=None, **kwargs):
    vmin, vmax = vmin, vmax

    def distr_maker(N, step=None):
        if step is not None and xi_max is not None and amp is not None:
            p0l = distr.cdf(xi_max - step/2, *args, **kwargs)
            p0r = distr.cdf(xi_max + step/2, *args, **kwargs)
            dp = (p0r - p0l) / N
            v_min = (vmin // step) * step
            v_max = (vmax // step) * step
            p1 = distr.cdf(v_min, *args, **kwargs)
            p2 = distr.cdf(v_max, *args, **kwargs)
            N = int(abs(p2-p1) // dp)
        else:
            if step is not None or xi_max is not None or amp is not None:
                print('Warning: Transverse distribution is generated.')
            p1 = 0 if vmin is None else distr.cdf(vmin, *args, **kwargs)
            p2 = 1 if vmax is None else distr.cdf(vmax, *args, **kwargs)
        return distr.ppf(np.linspace(p1, p2, N+2)[1:-1], *args, **kwargs)
    distr_maker.name = distr.name
    distr_maker.max = xi_max
    distr_maker.loc = args[-2]
    distr_maker.scale = args[-1]
    distr_maker.f = lambda x: distr.pdf(x, *args, **kwargs)
    if amp:
        distr_maker.amp = amp
    return distr_maker


def StepwiseXiDistr(distr, xi_max, vmin, vmax, amp=None):
    vmin, vmax = vmin, vmax

    def distr_maker(partic_in_layer, dxi):
        v_min = (vmin // dxi) * dxi
        v_max = (vmax // dxi) * dxi
        layers_borders = np.arange(v_max, v_min - dxi, -dxi)
        intervals = zip(layers_borders[1:], layers_borders)
        def N(xi): return int(partic_in_layer * distr(xi) / distr(xi_max))
        out = []
        for l, r in intervals:
            flat = stats.uniform.rvs(loc=l, scale=dxi, size=N((l+r)/2))
            out.append(flat)
        return np.hstack(out)
    distr_maker.name = 'arbitrary'
    distr_maker.max = xi_max
    distr_maker.f = distr
    if amp:
        distr_maker.amp = amp
    else:
        distr_maker.amp = distr(xi_max)
    return distr_maker


############ C config beam profile parsing ############
def find_beam_profile_pars(cfg):
    #WORD = '[a-z][a-z]*'
    #FLOAT = '[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'
    ans = re.findall(
        '([a-z][a-z]*/?[a-z]?)\s?=\s?([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?|[a-z][a-z]*)', cfg)
    return ans


def find_beam_profile(cfg):
    ans = re.search('beam-profile\s?=\s?"""([^\>]*)"""', cfg)
    return ans.group(1)


default_segment_pars = {'xi0': 0,
                        'xishape': 'cos',
                        'ampl': 0.5,
                        'length': 2*np.pi,
                        'rshape': 'g',
                        'vshift': 0,
                        'radius': 1,
                        'angshape': 'l',
                        'angspread': 1e-5,
                        'energy': 1000,
                        'vshift': 0,
                        'eshape': 'm',
                        'espread': 0,
                        'm/q': 1}


def split_into_segments(beam_profile_parsed):
    xi_current = 0
    segments = []
    segment = None
    for par in beam_profile_parsed:
        par_name, par_value = par[:2]
        if par_name == 'xishape':
            if segment:
                segments.append(segment)
                xi_current = xi_current - float(segment['length'])
            segment = copy(default_segment_pars)
            segment['xi0'] = xi_current
        if par_name.find('shape') > 0:
            segment[par_name] = par_value
        else:
            segment[par_name] = float(par_value)
    segments.append(segment)
    return segments


def get_segments_from_c_config(path):
    with open(path, 'r') as file:
        cfg = file.read()
    beam_profile = find_beam_profile(cfg)
    beam_profile_parsed = find_beam_profile_pars(beam_profile)
    segments = split_into_segments(beam_profile_parsed)
    return segments


def cosine(x, med, L): return 1./2 * (1. + np.cos(2. * np.pi * (x - med) / L))
def gauss(x, med, sigma): return np.exp(-(x-med)**2/2./sigma**2)


xishape_to_distr = dict(
    g=lambda xi0, length, amp: StepwiseXiDistr(
        lambda x: gauss(x, xi0, length/6.), xi0, xi0-length, xi0, amp),
    c=lambda xi0, length, amp: StepwiseXiDistr(
        lambda x: cosine(x, xi0, 2*length), xi0, xi0-length, xi0, amp),
    cos=lambda xi0, length, amp: StepwiseXiDistr(
        lambda x: cosine(x, xi0, 2*length), xi0, xi0-length, xi0, amp),
)
rshape_to_distr = dict(
    g=lambda radius: SmoothDistr(
        stats.weibull_min, 0, None, 2, 0, sqrt(2)*radius),
)
angshape_to_distr = dict(
    g=lambda angspread, energy: SmoothDistr(
        stats.norm, None, None, 0, energy*angspread),
    l=lambda angspread, energy: SmoothDistr(
        stats.norm, None, None, 0, energy*angspread),  # TODO
)
eshape_to_distr = dict(
    g=lambda erenergy, espread: SmoothDistr(
        stats.norm, None, None, erenergy, espread),
    m=lambda erenergy, espread: SmoothDistr(
        stats.norm, None, None, erenergy, 1e-15),
)


def distrs_from_shapes(segment, beam_current):
    distrs = dict(
        xi=xishape_to_distr[segment['xishape']](
            segment['xi0'], segment['length'], segment['ampl']*beam_current),
        r=rshape_to_distr[segment['rshape']](segment['radius']),
        p_z=eshape_to_distr[segment['eshape']](
            segment['energy'], segment['espread']),
        p_r=angshape_to_distr[segment['angshape']](
            segment['angspread'], segment['energy']),
        M=angshape_to_distr[segment['angshape']](
            segment['angspread'], segment['energy']),
    )
    return distrs
