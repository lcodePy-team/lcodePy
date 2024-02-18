import numpy as np
import pytest
import lcode
from lcode.plasma.data import Arrays

init_pl_2D_states = ([("./data/init_plasma/2D_uniform_particles_3.bin", 3),
                      ("./data/init_plasma/2D_uniform_particles_5.bin", 5),
                      ("./data/init_plasma/2D_uniform_particles_7.bin", 7),
                      ("./data/init_plasma/2D_uniform_particles_10.bin", 10),
                    ])

evol_pl_2D_states = ([("./data/plasma_solver/2D_state_0.npz"),
                      ("./data/plasma_solver/2D_state_1.npz"),
                      ("./data/plasma_solver/2D_state_2.npz"),
                      ("./data/plasma_solver/2D_state_Bz0_0.npz"),
                      ("./data/plasma_solver/2D_state_Bz0_1.npz"),
                      ("./data/plasma_solver/2D_state_Bz0_2.npz"),
                    ])

@pytest.fixture(scope='function')
def get_evol_config():
    config = {'geometry': '2d',
              'processing-2D_unit-type': 'cpu',
              'window-width-step-size': 0.01,
              'window-width': 2,
              'xi-step': 0.01,
              'plasma-particles-per-cell': 10,
             }
    return lcode.config.config.Config(config)



@pytest.mark.parametrize('path_to_reference, nppc', init_pl_2D_states)
def test_initialization(path_to_reference, nppc):
    """ test initial state of uniform plasma without ions """
    correct_2D_state = np.fromfile(path_to_reference, 
                                dtype="float64").reshape(6, -1)
    _config = {'geometry': '2d',
              'window-width-step-size': 0.01,
              'window-width': 1,
              'plasma-particles-per-cell': nppc}
    config = lcode.config.config.Config(_config)
    
    pl_2D_state = lcode.plasma.init_plasma_2d(config)
    fields = pl_2D_state[0]
    particles = pl_2D_state[1]
    currents = pl_2D_state[2]
    
    for fl in ("E_r", "E_f", "E_z", "B_f", "B_z"):
        assert np.all(getattr(fields, fl) == 0)
    for curr in ("rho", "j_r", "j_f", "j_z"):
        assert np.all(getattr(currents, curr) == 0)
    for i, attr in enumerate(("r", "p_r", "p_f", "p_z", "m", "q")):
        assert np.allclose(getattr(particles, attr), correct_2D_state[i, :], 
                           rtol=5e-16, atol=1e-125) 
     

@pytest.mark.parametrize('path_to_data', evol_pl_2D_states)
def test_plasma_decomposition(path_to_data, get_evol_config):
    conf = get_evol_config
    compute_rhoj = lcode.plasma.rhoj.get_rhoj_computer(conf)
    data = np.load(path_to_data)
    particles = Arrays(xp = np,
                       r = data["particles_prev"][0, :],
                       p_r = data["particles_prev"][1, :],
                       p_f = data["particles_prev"][2, :],
                       p_z = data["particles_prev"][3, :],
                       q = data["particles_prev"][4, :],
                       m = data["particles_prev"][5, :],
                       age = np.zeros_like(data["particles_prev"][0, :]),
                      )
    currents = compute_rhoj(particles)
    for i, attr in enumerate(("rho", "j_r", "j_f", "j_z")):
        assert np.allclose(getattr(currents, attr), 
                           data["currents_prev"][i,:], 
                           rtol=5e-11, atol=1e-125) 
     
@pytest.mark.parametrize('path_to_data', evol_pl_2D_states)
def test_fields_solver(path_to_data, get_evol_config):
    conf = get_evol_config
    fields_calculator = lcode.plasma.fields.get_field_computer(conf)
    data = np.load(path_to_data)
    fields_prev = Arrays(xp = np,
                         E_r = data["fields_prev"][0, :],
                         E_f = data["fields_prev"][1, :],
                         E_z = data["fields_prev"][2, :],
                         B_z = data["fields_prev"][3, :],
                         B_f = data["fields_prev"][4, :],
                        )
    fields_pred = Arrays(xp = np,
                         E_r = data["fields_pred"][0, :],
                         E_f = data["fields_pred"][1, :],
                         E_z = np.zeros_like(data["fields_pred"][0, :]),
                         B_z = np.zeros_like(data["fields_pred"][0, :]),
                         B_f = np.zeros_like(data["fields_pred"][0, :]),
                        )
    currents_prev = Arrays(xp = np,
                           rho = data["currents_prev"][0, :],
                           j_r = data["currents_prev"][1, :],
                           j_f = data["currents_prev"][2, :],
                           j_z = data["currents_prev"][3, :],
                          )
    currents_pred = Arrays(xp = np,
                           rho = data["currents_pred"][0, :],
                           j_r = data["currents_pred"][1, :],
                           j_f = data["currents_pred"][2, :],
                           j_z = data["currents_pred"][3, :],
                          )
    jbz = data["beam_current"]
    fields = fields_calculator(fields_pred, fields_prev, jbz, 
                               currents_prev, currents_pred, 0.01)[0]
    if "Bz0" in path_to_data:
        for i, attr in enumerate(("E_r", "E_f", "E_z", "B_z", "B_f")):
            assert np.allclose(getattr(fields, attr), 
                               data["fields"][i,:], 
                               rtol=1e-9, atol=1e-125) 
    else:
        for i, attr in enumerate(("E_r", "E_f", "E_z", "B_z", "B_f")):
            assert np.allclose(getattr(fields, attr), 
                               data["fields"][i,:], 
                               rtol=1e-9, atol=1e-125) 
        for i, attr in enumerate(("E_f", "B_z")):
            assert np.all(getattr(fields, attr) == 0)

@pytest.mark.parametrize('path_to_data', evol_pl_2D_states)
def test_plasma_pusher(path_to_data, get_evol_config):
    conf = get_evol_config
    if "Bz0" in path_to_data:
        conf.set("magnetic-field", 0.5)
    pusher = lcode.plasma.move.get_plasma_particles_mover(conf)
    data = np.load(path_to_data)
    E_r = (data["fields_prev"][0, :] + data["fields"][0, :]) / 2
    E_f = (data["fields_prev"][1, :] + data["fields"][1, :]) / 2
    E_z = (data["fields_prev"][2, :] + data["fields"][2, :]) / 2
    B_z = (data["fields_prev"][3, :] + data["fields"][3, :]) / 2
    B_f = (data["fields_prev"][4, :] + data["fields"][4, :]) / 2
    
    fields_aver = Arrays(xp = np,
                         E_r = E_r, 
                         E_f = E_f, 
                         E_z = E_z, 
                         B_z = B_z, 
                         B_f = B_f, 
                        )
    particles_prev = Arrays(xp = np,
                            r = data["particles_prev"][0, :],
                            p_r = data["particles_prev"][1, :],
                            p_f = data["particles_prev"][2, :],
                            p_z = data["particles_prev"][3, :],
                            q = data["particles_prev"][4, :],
                            m = data["particles_prev"][5, :],
                            age = np.zeros_like(data["particles_prev"][0, :]),
                           )
    particles = pusher(fields_aver, particles_prev, 
                       np.zeros_like(E_r), 0.01)
    for i, attr in enumerate(("r", "p_r", "p_f", "p_z", "q", "m")):
        assert np.allclose(getattr(particles, attr), 
                           data["particles"][i,:], 
                           rtol=1e-12, atol=1e-125) 
