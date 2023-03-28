# Import required modules
import numpy as np
import matplotlib.pyplot as plt
import time

from lcode2dPy.plasma3d.initialization import init_plasma
from lcode2dPy.plasma3d.solver import Plane2d3vPlasmaSolver

from lcode2dPy.config.config import Config


def rho_b_test1(xi, x_grid, y_grid):
    COMPRESS, BOOST, SIGMA, SHIFT = 1, 1, 1, 0
    if xi < -2 * np.sqrt(2 * np.pi) / COMPRESS or xi > 0:
        return 0.
        # return np.zeros_like(xs)
    r = np.sqrt(x_grid**2 + (y_grid - SHIFT)**2)
    return (.05 * BOOST * np.exp(-.5 * (r / SIGMA)**2) *
            (1 - np.cos(xi * COMPRESS * np.sqrt(np.pi / 2))))


def calculate_head(window_length: float, config_dictionary: dict,
                   rho_b: rho_b_test1):
    # Creates a config
    config = Config(config_dictionary)

    # Makes ready a plasma solver
    solver = Plane2d3vPlasmaSolver(config)

    # Creates a transversal grid
    grid_steps     = config.getint('window-width-steps')
    grid_step_size = config.getfloat('window-width-step-size')

    grid = ((np.arange(grid_steps) - grid_steps // 2)
            * grid_step_size)
    x_grid, y_grid = grid[:, None], grid[None, :]

    # Creates the first plasma slice
    fields, particles, currents, const_arrays = init_plasma(config)

    # Simulation loop:
    start_time = time.time()
    xi_step_size = config.getfloat('xi-step')
    xi_steps = round(window_length / xi_step_size)

    for xi_i in range(xi_steps + 1):
        xi = - xi_i * xi_step_size

        ro_beam = rho_b(xi, x_grid, y_grid)
        ro_beam_prev = rho_b(xi + xi_step_size, x_grid, y_grid)

        particles, fields, currents = solver.step_dxi(
            particles, fields, currents, const_arrays, ro_beam, ro_beam_prev
        )

        ez = fields.Ez[grid_steps // 2, grid_steps // 2]
        if xi_i % 1. == 0:
            print(f'xi={xi:+.4f} Ez={ez:+.4e}')

    np.savez_compressed(
        file='plasmastate_after_the_head',
        x_init=particles.x_init, y_init=particles.y_init,
        x_offt=particles.x_offt, y_offt=particles.y_offt,
        px=particles.px, py=particles.py, pz=particles.pz,
        q=particles.q, m=particles.m,
        
        dx_chaotic=particles.dx_chaotic,
        dy_chaotic=particles.dy_chaotic,
        dx_chaotic_perp=particles.dx_chaotic_perp,
        dy_chaotic_perp=particles.dy_chaotic_perp,

        Ex=fields.Ex, Ey=fields.Ey, Ez=fields.Ez,
        Bx=fields.Bx, By=fields.By, Bz=fields.Bz,

        Phi=fields.Phi, ro=currents.ro,
        jx=currents.jx, jy=currents.jy, jz=currents.jz
    )

    print("--- %s seconds spent calculating only the head ---" % (time.time() - start_time))
