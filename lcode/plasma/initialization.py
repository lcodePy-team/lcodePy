"""Module for plasma initialization routines."""
import numpy as np

from ..config.config import Config
from .data import Arrays
from .rhoj import get_rhoj_computer
from ..plasma3d.profile import profile_initial_plasma

def _place_particles(window_width, r_step, particles_per_cell):
    n_particles = int(window_width / r_step) * particles_per_cell
    r_step_p = window_width / n_particles
    return (np.arange(n_particles) + 0.5) * r_step_p

def _weight_particles(particle_positions):
    r_step_p = particle_positions[1] - particle_positions[0]
    return 2 * np.pi * r_step_p * particle_positions

def init_plasma(config: Config, current_time=0):
    window_width = config.getfloat('window-width')
    r_step = config.getfloat('transverse-step')
    xi_step = config.getfloat('xi-step')
    part_per_cell = config.getint('plasma-particles-per-cell')
    path_lim = config.getfloat('trapped-path-limit')
    ion_model = config.get("ion-model")
    grid_length = int(window_width / r_step) + 1

    r_p = _place_particles(window_width, r_step, part_per_cell)
    m_p = _weight_particles(r_p)
    q_p = -np.copy(m_p)
    q_p, m_p  = profile_initial_plasma(config, current_time, r_p, 
                                       np.zeros_like(r_p), q_p, m_p)
    mask = q_p != 0
    q_p = q_p[mask].copy()
    m_p = m_p[mask].copy()
    r_p = r_p[mask].copy()
    
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

    xi_plasma_layer_start = xi_step
    return fields, particles, currents, const_arrays, xi_plasma_layer_start


def load_plasma(config: Config, path_to_plasmastate: str):
    fields, particles, currents, const_arrays, xi_plasma_layer =\
        init_plasma(config)
    xp = fields.xp
    with np.load(file=path_to_plasmastate) as state:
        fields = Arrays(xp=xp,
                        E_r=state['E_r'], E_f=state['E_f'], E_z=state['E_z'],
                        B_f=state['B_f'], B_z=state['B_z'])
        for sort in const_arrays.sorts:
            particles[sort] = Arrays(
                xp=xp,
                r=state[sort]['r'], q=state[sort]['q'], m=state[sort]['m'], 
                p_r=state[sort]['p_r'], p_f=state[sort]['p_f'], 
                p_z=state[sort]['p_z'], age=state[sort]['age'],
            )

        currents = Arrays(xp=xp,
                          rho=xp.array(state['rho']), j_r=xp.array(state['j_r']),
                          j_f=xp.array(state['j_f']), j_z=xp.array(state['j_z']))

        try:
            const_arrays.ni = state['ni']
        except:
            pass
        try: 
            xi_plasma_layer = float(state['xi_plasma_layer'])
        except:
            xi_plasma_layer = 0

    return fields, particles, currents, const_arrays, xi_plasma_layer
