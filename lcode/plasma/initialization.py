"""Module for plasma initialization routines."""
import numpy as np

from ..config.config import Config
from .data import Arrays
from .profiles import get_plasma_profile
from .rhoj import get_rhoj_computer


def init_plasma(config: Config, current_time=0):
    window_width = config.getfloat('window-width')
    r_step = config.getfloat('transverse-step')
    part_per_cell = config.getint('plasma-particles-per-cell')
    path_lim = config.getfloat('trapped-path-limit')
    ion_model = config.get("ion-model")
    grid_length = int(window_width / r_step) + 1

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

    # A short function that creates a numpy array of zeros. We need it so we
    # don't face the problem of views of numpy arrays.
    def zeros(size=1):
        if size == 1:
            return np.zeros(grid_length, dtype=np.float64)
        else:
            return np.zeros(shape=(size,grid_length), dtype=np.float64)


    fields = Arrays(xp=np, E_r=zeros(), E_f=zeros(), 
                    E_z=zeros(), B_f=zeros(), B_z=zeros())
    

    particles = {'electrons' : 
                 Arrays(xp=np, r=r_p, p_r=p_r_p, p_f=p_f_p, 
                        p_z=p_z_p, q=q_p, m=m_p, age=age)
                }
    
    currents = Arrays(xp=np, rho=zeros(2), j_r=zeros(2), 
                      j_f=zeros(2), j_z=zeros(2))

    if  ion_model == "background":       
        const_arrays = Arrays(xp=np, ni=zeros())
        const_arrays.sorts = {'electrons' : 0}
        compute_rhoj = get_rhoj_computer(config)
        ne = compute_rhoj(particles, const_arrays).rho[0, :]
        currents.rho[0, :] = ne[:]
        currents.rho[1, :] = -ne[:]
        const_arrays.ni = -ne[:]
    elif ion_model == "mobile":       
        ion_mass = config.getint("ion-mass")
        r_p = r_p.copy()
        q_p = -q_p.copy()
        m_p = ion_mass * q_p.copy()
        p_r_p = np.zeros_like(r_p)
        p_f_p = np.zeros_like(r_p)
        p_z_p = np.zeros_like(r_p)
        age = np.zeros_like(r_p)
        particles['ions']= Arrays(xp=np, r=r_p, p_r=p_r_p, p_f=p_f_p, 
                                    p_z=p_z_p, q=q_p, m=m_p, age=age)

        const_arrays = Arrays(xp=np)
        const_arrays.sorts = {'electrons' : 0, 'ions' : 1}

    
    return fields, particles, currents, const_arrays


def load_plasma(config: Config, path_to_plasmastate: str):
    fields, particles, currents = init_plasma(config)

    with np.load(file=path_to_plasmastate) as state:
        fields = Arrays(xp=np, E_r=state['Er'], E_f=state['Ef'],
                        E_z=state['Ez'], B_f=state['Bf'], B_z=state['Bz'])

        particles = Arrays(xp=np, r=state['r'],
                           p_r=state['pr'], p_f=state['pf'], p_z=state['pz'],
                           q=state['q'], m=state['m'], age=state['age'])

        currents = Arrays(xp=np, rho=state['ro'],
                          j_x=state['jx'], j_y=state['jy'], j_z=state['jz'])

    return fields, particles, currents
