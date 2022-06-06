import numpy as np
from numba import njit
from lcode2dPy.beam import (
    beam_slice_mover
)
from mpi4py import MPI
from lcode2dPy.beam.beam_calculate import layout_beam_slice
from lcode2dPy.beam.beam_slice import BeamSlice
from lcode2dPy.plasma.solver import CylindricalPlasmaSolver

particle_dtype = np.dtype(
    [
        ('xi', 'f8'),
        ('r', 'f8'),
        ('p_z', 'f8'),
        ('p_r', 'f8'),
        ('M', 'f8'),
        ('q_m', 'f8'),
        ('q_norm', 'f8'),
        ('id', 'i8'),
    ],
)


# split beam slice into lost, stable and moving beam slices
@njit
def split_beam_slice(beam_slice, xi_end):
    lost_slice = beam_slice[beam_slice.lost]
    temp_slice = beam_slice[np.invert(beam_slice.lost)]
    moving_mask = np.logical_or(temp_slice.remaining_steps > 0,
                                temp_slice.xi < xi_end)
    moving_slice = temp_slice[moving_mask]
    stable_slice = temp_slice[np.invert(moving_mask)]

    return lost_slice, stable_slice, moving_slice


class PusherAndSolver:
    def __init__(self, config):
        self.config = config
        # self.layout_beam_slice = configure_layout_beam_slice(config)
        self.move_beam_slice = beam_slice_mover(config)
        self.solver = CylindricalPlasmaSolver(config)
        self.r_step = float(config.get('window-width-step-size'))
        self.is_rigid = 1 if config.get('rigid-beam') == 'y' else 0
        max_radius = float(config.get('window-width'))
        self.n_cells = int(max_radius / self.r_step) + 1
        self.xi_step_p = config.getfloat('xi-step')
        self.window_length = config.getfloat('window-length')
        self.xi_layers_num = int(self.window_length / self.xi_step_p)

    def step_dt(self, plasma_particles, plasma_fields,
                beam_source, beam_drain, t, diagnostics):
        diagnostics.before(t)

        beam_slice_to_move = beam_source.get_beam_slice(0, -self.xi_step_p)
        rho_layout = np.zeros(self.n_cells)
        for layer_idx in np.arange(self.xi_layers_num):
            # Get beam layer with xi \in [xi^{layer_idx + 1}, xi^{layer_idx})
            # Its index is `layer_idx`
            if layer_idx == self.xi_layers_num - 1:
                beam_slice_to_layout = BeamSlice(0)
            else:
                beam_slice_to_layout = beam_source.get_beam_slice(
                    (layer_idx + 1) * -self.xi_step_p,
                    (layer_idx + 2) * -self.xi_step_p,
                )
            rho_beam, rho_layout = layout_beam_slice(
                beam_slice_to_move.concat(beam_slice_to_layout),
                layer_idx,
                rho_layout,
                self.r_step,
                self.xi_step_p,
            )

            # Now we can compute plasma layer `layer_idx` reaction
            plasma_particles_new, plasma_fields_new, steps = \
                self.solver.step_dxi(
                    plasma_particles, plasma_fields, rho_beam,
                )
            # Now we can move beam layer `layer_idx - 1`

            self.move_beam_slice(
                beam_slice_to_move,
                layer_idx + 1,
                plasma_fields_new,
                plasma_fields,
            )
            # TODO lost потом обрабатывать
            lost_slice, stable_slice, moving_slice = split_beam_slice(
                beam_slice_to_move, (layer_idx + 1) * -self.xi_step_p,
            )

            beam_slice_to_move = beam_slice_to_layout.concatenate(moving_slice)
            plasma_particles = plasma_particles_new
            plasma_fields = plasma_fields_new

            diagnostics.dxi(t, layer_idx, plasma_particles,
                            plasma_fields, rho_beam, stable_slice)

            beam_drain.push_beam_slice(stable_slice)
            beam_drain.finish_layer((layer_idx + 1) * -self.xi_step_p)

            with open(f'worker_{MPI.COMM_WORLD.rank:04}', 'w') as f:
                f.write(f"t: {t}, xi: {layer_idx * self.xi_step_p}")

        diagnostics.after()
        return plasma_particles, plasma_fields
