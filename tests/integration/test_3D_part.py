from os import path
import pytest
import numpy as np
import lcode

DATA_DIR = path.join(path.dirname(path.abspath(__file__)), 'data')
RTOL = 5e-8

@pytest.fixture(scope='function')
def get_evol_config():
    return { 'geometry': '3d',
             'processing-unit-type': 'cpu',

             'window-width': 16,
             'window-width-step-size': 0.05,

             'window-length': 10, 
             'xi-step': 0.05,

             'time-limit': 1,
             'time-step': 1,

             'plasma-particles-per-cell': 4,
           }

    
     

# TODO rigid beam to avoid random seed
def test_test1(get_evol_config):

    config = get_evol_config
    default = {'length' : 5.013256548}
    beam_parameters = {'current': 0.05 * (2*3.14), 'particles_in_layer': 5000, 
                       'default' : default} 
    diags = [] 
    sim = lcode.Simulation(config=config, diagnostics=diags,
                                 beam_parameters=beam_parameters)
   
    sim.step()
    particles, fields, currents = sim._Simulation__push_solver._plasmastate

    result = np.load(path.join(DATA_DIR, "3D_test1.npz"))

    for attr in ("x_offt", "y_offt", "px", "py", "pz"):
        assert np.allclose(getattr(particles, attr)[::10, ::10], result[attr], 
                           rtol=RTOL, atol=1e-125)
    for attr in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
        assert np.allclose(getattr(fields, attr)[::10, ::10], result[attr], 
                           rtol=RTOL, atol=1e-125)
    for attr in ("ro", "jx", "jy", "jz"):
        assert np.allclose(getattr(currents, attr)[::10, ::10], result[attr], 
                           rtol=RTOL, atol=1e-125)


def test_beam_evol(get_evol_config):

    config = get_evol_config
    config["window-length"] = 5
    config["time-limit"] = 5
    config["enable-noise-filter"] = False
    beam_parameters = {'current': 0.5 * (2*3.14), 'particles_in_layer': 1000} 
    diags = [] 
    sim = lcode.Simulation(config=config, diagnostics=diags,
                                 beam_parameters=beam_parameters)
    sim.step()
    particles, fields, currents = sim._Simulation__push_solver._plasmastate

    result = np.load(path.join(DATA_DIR, "3D_beam_evol.npz"))

    for attr in ("x_offt", "y_offt", "px", "py", "pz"):
        assert np.allclose(getattr(particles, attr)[::10, ::10], result[attr], 
                           rtol=RTOL, atol=1e-125)
    for attr in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
        assert np.allclose(getattr(fields, attr)[::10, ::10], result[attr], 
                           rtol=RTOL, atol=1e-125)
    for attr in ("ro", "jx", "jy", "jz"):
        assert np.allclose(getattr(currents, attr)[::10, ::10], result[attr], 
                           rtol=RTOL, atol=1e-125)
