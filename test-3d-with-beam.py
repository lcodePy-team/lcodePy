from copy import copy

import numpy as np
import time

from lcode2dPy.plasma3d.initialization import init_plasma
from lcode2dPy.push_solver_3d import PushAndSolver3d

from lcode2dPy.beam3d import beam, beam_generator

from lcode2dPy.config.default_config_3d import default_config

start_time = time.time()
config = copy(default_config)

config.set('window-xy-steps', 769)
config.set('window-xy-step-size', 0.02)
config.set('window-length', 15)
config.set('xi-step', 0.02)

config.set('reflect-padding-steps', 10)
config.set('plasma-padding-steps', 10)
config.set('plasma_fineness', 2)

pusher_solver = PushAndSolver3d(config)

time_limit = config.getfloat('time_limit')
time_step_size  = config.getfloat('time_step')
time_steps = int(time_limit / time_step_size)

beam_generator.main()

time.sleep(5.)

beam_particles = beam.BeamParticles(0)
beam_particles.load('beamfile.npz')
beam_calulator = beam.BeamCalculator(config, beam_particles)

for t_i in range(time_steps + 1):
    pl_fields, pl_particles, pl_currents, pl_const_arrays = init_plasma(config)

    pusher_solver.step_dt(pl_fields, pl_particles, pl_currents, pl_const_arrays,
                          beam_particles, beam_calulator)

print("--- %s seconds ---" % (time.time() - start_time))