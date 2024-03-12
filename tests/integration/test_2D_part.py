from os import path
import pytest
import numpy as np
import lcode

DATA_DIR = path.join(path.dirname(path.abspath(__file__)), 'data')
RTOL = 5e-8

@pytest.fixture(scope='function')
def get_evol_config():
    return {
            'geometry': '2d',
            'processing-unit-type': 'cpu',
            
            'window-width-step-size': 0.01,
            'window-width': 16,

            'window-length': 10, 
            'xi-step': 0.01,

            'time-limit': 1,
            'time-step': 1,

            'plasma-particles-per-cell': 10,
            'noise-reductor-enabled' : False,
            
           }

    
     

# TODO rigid beam to avoid random seed
def test_test1(get_evol_config):

    config = get_evol_config
    config["ion-model"] = 'background'
    default = {'length' : 5.013256548}
    beam_parameters = {'current': 0.05, 'particles_in_layer': 5000, 
                       'default' : default} 
    diags = [] 
    sim = lcode.Simulation(config=config, diagnostics=diags,
                                 beam_parameters=beam_parameters)
   
    sim.step()
    particles, fields, currents = sim._Simulation__push_solver._plasmastate
    particles = particles["electrons"]

    result = np.load(path.join(DATA_DIR, "2D_test1.npz"))

    for attr in ("r", "p_r", "p_f", "p_z"):
        assert np.allclose(getattr(particles, attr)[::50], result[attr], 
                           rtol=RTOL, atol=1e-125)
    for attr in ("E_r", "E_f", "E_z", "B_f", "B_z"):
        assert np.allclose(getattr(fields, attr)[::50], result[attr], 
                           rtol=RTOL, atol=1e-125)
    for attr in ("rho", "j_r", "j_f", "j_z"):
        assert np.allclose(getattr(currents, attr)[0,::50], result[attr], 
                           rtol=RTOL, atol=1e-125)


def test_beam_evol(get_evol_config):

    config = get_evol_config
    config["time-limit"] = 3
    beam_parameters = {'current': 0.5, 'particles_in_layer': 300} 
    diags = [] 
    sim = lcode.Simulation(config=config, diagnostics=diags,
                                 beam_parameters=beam_parameters)
    sim.step()
    particles, fields, currents = sim._Simulation__push_solver._plasmastate
    particles = particles["electrons"]

    result = np.load(path.join(DATA_DIR, "2D_beam_evol.npz"))

    for attr in ("r", "p_r", "p_f", "p_z"):
        assert np.allclose(getattr(particles, attr)[::50], result[attr], 
                           rtol=RTOL, atol=1e-125)
    for attr in ("E_r", "E_f", "E_z", "B_f", "B_z"):
        assert np.allclose(getattr(fields, attr)[::50], result[attr], 
                           rtol=RTOL, atol=1e-125)
    for attr in ("rho", "j_r", "j_f", "j_z"):
        assert np.allclose(getattr(currents, attr)[0,::50], result[attr], 
                           rtol=RTOL, atol=1e-125)
