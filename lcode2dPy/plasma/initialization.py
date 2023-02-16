"""Module for plasma initialization routines."""
import numpy as np

from ..config.config import Config
from .data import Fields, Particles, Currents
from .profiles import get_plasma_profile


def init_plasma(config: Config, current_time=0):
    window_width = config.getfloat('window-width')
    r_step = config.getfloat('window-width-step-size')
    part_per_cell = config.getint('plasma-particles-per-cell')
    path_lim = config.getfloat('trapped-path-limit')
    grid_length = int(window_width / r_step) + 1
    fields = Fields(grid_length)
    plasma_profile = get_plasma_profile(config)
    r_p = plasma_profile.place_particles(part_per_cell)
    m_p = plasma_profile.weigh_particles(r_p)
    q_p = -np.copy(m_p)
    p_r_p = np.zeros_like(r_p)
    p_f_p = np.zeros_like(r_p)
    p_z_p = np.zeros_like(r_p)
    if path_lim > 0:
        age = np.full_like(r_p, path_lim)
    else:
        age = np.zeros_like(r_p)

    zeros = np.zeros(grid_length, dtype=np.float64)
    return (fields, Particles(r_p, p_r_p, p_f_p, p_z_p, q_p, m_p, age),
            Currents(rho=zeros, j_r=zeros, j_f=zeros, j_z=zeros))


def load_plasma(config: Config, path_to_plasmastate: str):
    with np.load(file=path_to_plasmastate) as state:
        fields = Fields(E_r=state['Er'], E_f=state['Ef'], E_z=state['Ez'],
                        B_f=state['Bf'], B_z=state['Bz'])

        particles = Particles(r=state['r'], p_r=state['pr'], p_f=state['pf'],
                              p_z=state['pz'], q=state['q'], m=state['m'],
                              age=state['age'])

        currents = Currents(rho=state['ro'], j_x=state['jx'],
                            j_y=state['jy'], j_z=state['jz'])

    return fields, particles, currents
