from numpy.lib.twodim_base import diag
from lcode2dPy.simulation.interface import Simulation
from lcode2dPy.diagnostics.targets import BeamDiagnostics, FieldDiagnostics, PlasmaDiagnostics
from lcode2dPy.config.default_config import default_config
from lcode2dPy.beam_generator.beam_generator import make_beam, Gauss, rGauss
import numpy as np
import pickle
import subprocess
import os
try:
    os.remove('beamfile.bin')
except:
    pass


time_step       = 200
time_limit      = 200.5
window_length   = 20
window_width    = 2
r_step          = 0.01
xi_step         = 0.01

config = default_config
config.set('time-step', time_step)
config.set('time-limit', time_limit)
config.set('window-length', window_length)
config.set('window-width', window_width)
config.set('window-width-step-size', r_step)
config.set('xi-step', xi_step)

# Beam
gamma = 426
angspread = 1e-5
m_proton = 958/0.51

beam_pars = dict(xi_distr=Gauss(sigma=100, vmin=-window_length, vmax=0),
                r_distr=rGauss(vmin=0, vmax=window_width),
                ang_distr=Gauss(sigma=angspread, vmin=None, vmax=None),
                pz_distr=Gauss(gamma*m_proton, gamma*m_proton*1e-4, vmin=None, vmax=None),
                Ipeak_kA=20*40/1000,
                q_m=1/m_proton,
                saveto=".")

diagnostics = [
    # BeamDiagnostics(period=time_limit//time_step * time_step),
    # PlasmaDiagnostics(period=time_limit//time_step * time_step)
    FieldDiagnostics(name="E_z", period=time_limit//time_step * time_step)
]

sim = Simulation(beam_pars=beam_pars, diagnostics=diagnostics, config=config)

sim.step(int(time_limit//time_step))

print("end")

with open("diagnostics.pickle", "wb") as f:
    pickle.dump(diagnostics, f)