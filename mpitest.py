import os
from lcode2dPy.diagnostics.openpmd import Diagnostics
# FieldsDiagnostics, PlasmaDiagnostics, BeamDiagnostics
from lcode2dPy.diagnostics.openpmd import BeamDiagnostics
from lcode2dPy.simulation.interface import Simulation
from lcode2dPy.config.default_config import default_config
from lcode2dPy.beam.beam_generator import Gauss, rGauss

from numba import set_num_threads
set_num_threads(1)

time_step = 500
time_limit = 10000
window_length = 15
window_width = 4
r_step = 0.05
xi_step = 0.05

try:
    os.remove('./beamfile.bin')
except:
    pass

config = default_config
config.set('time-step', time_step)
config.set('time-limit', time_limit)
config.set('window-length', window_length)
config.set('window-width', window_width)
config.set('window-width-step-size', r_step)
config.set('xi-step', xi_step)

gamma = 426
angspread = 1e-5
m_proton = 958/0.51

beam_pars = dict(xi_distr=Gauss(sigma=100, vmin=-window_length, vmax=0),
                 r_distr=rGauss(vmin=0, vmax=window_width),
                 ang_distr=Gauss(sigma=angspread, vmin=None, vmax=None),
                 pz_distr=Gauss(gamma*m_proton, gamma*m_proton*1e-4,
                                vmin=None, vmax=None),
                 Ipeak_kA=2*40/1000,
                 q_m=1/m_proton,
                 saveto=".")

diagnostics = Diagnostics(
    path="./diagnostics/",
    author="Nikita Okhotnikov <nikiquark@gmail.com>",
    content=[
        # FieldsDiagnostics(config, period=200, time=(None,None), select={"E_z": -1}),
        # PlasmaDiagnostics(config, period=200, time=(None,None)),
        BeamDiagnostics(config, period=200, time=(None, None))
    ]
)

sim = Simulation(beam_pars=beam_pars, diagnostics=diagnostics, config=config)

sim.step(int(time_limit//time_step))
