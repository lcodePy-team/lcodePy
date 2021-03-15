from copy import copy

import numpy as np
import time

from lcode2dPy.plasma3d.initialization import init_plasma
from lcode2dPy.plasma3d.solver import Plane2d3vPlasmaSolver 

from lcode2dPy.config.default_config import default_config

start_time = time.time()
config = copy(default_config)

config.set('window-width-steps', 769)
config.set('window-width-step-size', 0.02)
config.set('window-length', 10)
config.set('xi-step', 0.02)

config.set('reflect-padding-steps', 10)
config.set('plasma-padding-steps', 10)
config.set('plasma_fineness', 2)


solver = Plane2d3vPlasmaSolver(config)

grid_steps     = config.getint('window-width-steps')
grid_step_size = config.getfloat('window-width-step-size')
xi_step_size   = config.getfloat('xi-step')

grid = ((np.arange(grid_steps) - grid_steps // 2)
        * grid_step_size)
xs, ys = grid[:, None], grid[None, :]

def rho_b(xi):
    COMPRESS, BOOST, SIGMA, SHIFT = 1, 5, 1, 0
    if xi < -2 * np.sqrt(2 * np.pi) / COMPRESS:
        return 0.
        # return np.zeros_like(xs)
    r = np.sqrt(xs**2 + (ys - SHIFT)**2)
    return (.05 * BOOST * np.exp(-.5 * (r / SIGMA)**2) *
            (1 - np.cos(xi * COMPRESS * np.sqrt(np.pi / 2))))

xi = 0
fields, particles, currents, const_arrays = init_plasma(config)
xi_max = config.getfloat('window-length')

i = 0
ez = np.zeros(int(xi_max / xi_step_size) + 1)
while xi > -xi_max:
# for _ in range(1):
    rho_b_ = rho_b(xi)
    particles, fields, currents = solver.step_dxi(particles, fields, currents, 
                                                  const_arrays, rho_b_)
    # print(fields.E_z[grid_steps // 2, grid_steps // 2])
    ez[i] = fields.E_z[grid_steps // 2, grid_steps // 2]
    # print('xi={xi:.6f} Ez={Ez:e}'.format(xi=xi, Ez=fields.E_z[grid_steps // 2, grid_steps // 2]))
    print(f'xi={xi:+.4f} {ez[i]:+.4e}')
    i += 1
    xi -= xi_step_size
print("--- %s seconds ---" % (time.time() - start_time))
np.save('ez.npy', ez)
