import cupy as cp

from lcode2dPy.config.config import Config
from lcode2dPy.plasma3d_gpu.solver import Plane2d3vPlasmaSolver
from lcode2dPy.plasma3d_gpu.data import GPUArrays, GPUArraysView
from lcode2dPy.beam3d_gpu.beam import (BeamParticles, BeamSource, BeamDrain,
                                       BeamCalculator, concatenate_beam_layers)
from lcode2dPy.diagnostics.diagnostics_3d import Diagnostics3d


class PushAndSolver3d:
    def __init__(self, config: Config):
        self.config = config

        # Import plasma solver and beam pusher, pl = plasma
        self.pl_solver = Plane2d3vPlasmaSolver(config)
        self.beam_calc = BeamCalculator(config)

        self.xi_max = config.getfloat('window-length')
        self.xi_step_size = config.getfloat('xi-step')
        self.xi_steps = int(self.xi_max / self.xi_step_size)
        self.grid_steps = config.getint('window-width-steps')

    def step_dt(self, pl_fields: GPUArrays, pl_particles: GPUArrays,
                pl_currents: GPUArrays, pl_const_arrays: GPUArrays,
                beam_source: BeamSource, beam_drain: BeamDrain,
                current_time, diagnostics: Diagnostics3d=None):
        """
        Perform one time step of beam-plasma calculations.
        """
        self.beam_calc.start_time_step()
        beam_layer_to_move = BeamParticles(0)
        fell_size = 0

        gpu_index = 0
        with cp.cuda.Device(gpu_index):
            for xi_i in range(self.xi_steps + 1):
                beam_layer_to_layout = beam_source.get_beam_layer_to_layout(xi_i)
                ro_beam = (
                    self.beam_calc.layout_beam_layer(beam_layer_to_layout, xi_i))

                prev_pl_fields = pl_fields.copy()

                pl_particles, pl_fields, pl_currents = self.pl_solver.step_dxi(
                    pl_particles, pl_fields, pl_currents, pl_const_arrays, ro_beam)

                lost, moved, fell_to_next_layer = self.beam_calc.move_beam_layer(
                    beam_layer_to_move, fell_size, xi_i, pl_fields, prev_pl_fields)

                # Beam layers operations:
                beam_layer_to_move = concatenate_beam_layers(beam_layer_to_layout,
                                                            fell_to_next_layer)
                fell_size = fell_to_next_layer.size

                beam_drain.push_beam_layer(moved)
                # beam_drain.push_beam_lost(lost)

                # Diagnostics:
                if diagnostics:
                    xi_plasma_layer = - self.xi_step_size * xi_i
                    diagnostics.dxi(current_time, xi_plasma_layer,
                        GPUArraysView(pl_particles), GPUArraysView(pl_fields),
                        GPUArraysView(pl_currents), ro_beam.get())

                # # Some diagnostics:
                view_pl_fields = GPUArraysView(pl_fields)
                Ez_00 = view_pl_fields.Ez[self.grid_steps//2, self.grid_steps//2]

                print(f'xi={-xi_i * self.xi_step_size:+.4f} {Ez_00:+.4e}')

        if diagnostics:
            diagnostics.dump(current_time)
