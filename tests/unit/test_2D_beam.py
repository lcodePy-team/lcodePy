from os import path
import numpy as np
import pytest
import lcode
from lcode.plasma.data import Arrays

DATA_DIR = path.join(path.dirname(path.abspath(__file__)), 'data', 'beam_solver')

init_data_2D = ([(path.join(DATA_DIR, "2D_state_1.npz"), 
                   {}),
                 (path.join(DATA_DIR, "2D_state_2.npz"),
                   {"magnetic-field" : 5}),
                 (path.join(DATA_DIR, "2D_state_3.npz"),
                   {"beam-substepping-energy" : 3900}),
               ])

@pytest.fixture(scope='function')
def get_evol_config_2D():
    config = {'geometry': '2d',
              'processing-unit-type': 'cpu',
              'time-step' : 40,
              'window-width-step-size': 0.01,
              'window-width': 2,
              'xi-step': 0.01,
              'plasma-particles-per-cell': 10,
              'beam-substepping-energy' : 2,
             }
    return lcode.config.config.Config(config)


def test_memory_beam_source(get_evol_config_2D):
    conf = get_evol_config_2D
    data = np.load(path.join(DATA_DIR, "2D_state_1.npz"))
    beam_layer = data["init_layer"]
    source = lcode.beam.beam_io.MemoryBeamSource(conf, beam_layer)
    sizes = [6, 106, 403]
    xi_i = 55
    for i in range(3):
        beam_layer_to_layout = source.get_beam_slice(-(xi_i+i) * 0.01, 
                                                -((xi_i+i) + 1.0001) * 0.01)
        assert(beam_layer_to_layout.size == sizes[i])

@pytest.mark.parametrize('path_to_data, extra_conf', init_data_2D)
def test_beam_decomposition(path_to_data, extra_conf, get_evol_config_2D):
    conf = get_evol_config_2D
    data = np.load(path_to_data)
    beam_layer = data["init_layer"]
    source = lcode.beam.beam_io.MemoryBeamSource(conf, beam_layer)
    beam_calc = lcode.beam.beam_calculate.BeamCalculator2D(conf)
    beam_calc.start_time_step()
    nb = np.zeros(shape = (201, 3))
    xi_i = 55
    for i in range(3):
        beam_layer_to_layout = source.get_beam_slice(-(xi_i+i) * 0.01, 
                                              -((xi_i+i) + 1.0001) * 0.01)
        nb[:, i] = beam_calc.layout_beam_layer(beam_layer_to_layout, xi_i+i)


    assert np.allclose(nb, data["density"], rtol=5e-16, atol=1e-125) 
     
@pytest.mark.parametrize('path_to_data, extra_conf', init_data_2D)
def test_beam_pusher(path_to_data, extra_conf, get_evol_config_2D):
    conf = get_evol_config_2D
    for key in extra_conf:
        conf.set(key, extra_conf[key])
    data = np.load(path_to_data)
    beam_layer = data["init_layer"]
    beam_layer = lcode.beam.data.BeamParticles(beam_layer.size, beam_layer)
    beam_layer.remaining_steps = data["init_remaining_steps"]
    beam_layer.dt = data["init_dt"]
    beam_calc = lcode.beam.beam_calculate.BeamCalculator2D(conf)
    beam_calc.start_time_step()
    prev_pl_fields = Arrays(xp=np, 
                            E_r=data["prev_pl_fields"][:,0],
                            E_f=data["prev_pl_fields"][:,1],
                            E_z=data["prev_pl_fields"][:,2],
                            B_f=data["prev_pl_fields"][:,3],
                            B_z=data["prev_pl_fields"][:,4],)
    pl_fields = Arrays(xp=np, 
                       E_r=data["pl_fields"][:,0],
                       E_f=data["pl_fields"][:,1],
                       E_z=data["pl_fields"][:,2],
                       B_f=data["pl_fields"][:,3],
                       B_z=data["pl_fields"][:,4],)
    fell_size = 0
    xi_i=58
    lost, moved, fell = beam_calc.move_beam_layer(beam_layer, 
                                               fell_size, xi_i, prev_pl_fields, 
                                               pl_fields)
    for attr in ("xi", "r", "p_r", "M", "p_z", "id"):
        assert np.allclose(lost.particles[attr], data["lost"][attr], 
                           rtol=5e-16, atol=1e-125)
    for attr in ("xi", "r", "p_r", "M", "p_z", "id"):
        assert np.allclose(moved.particles[attr], data["moved"][attr], 
                           rtol=5e-16, atol=1e-125)
    for attr in ("xi", "r", "p_r", "M", "p_z", "id"):
        assert np.allclose(fell.particles[attr], data["fell"][attr], 
                           rtol=5e-16, atol=1e-125)
