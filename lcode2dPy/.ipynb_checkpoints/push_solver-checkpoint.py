import numpy as np
from numba import njit

from lcode2dPy.beam import (
    BeamSlice,
    beam_slice_mover,
    #layout_beam_slice,
)
from lcode2dPy.beam.beam_calculate import layout_beam_slice
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
# @nb.njit
def split_beam_slice(beam_slice, xi_end):
    lost_slice = BeamSlice(0)

    moving_mask = np.logical_or(beam_slice.remaining_steps > 0, beam_slice.xi < xi_end)
    stable_count = moving_mask.size - np.sum(moving_mask)

    sorted_idxes = np.argsort(moving_mask)
    beam_slice.particles = beam_slice.particles[sorted_idxes]
    beam_slice.dt = beam_slice.dt[sorted_idxes]
    beam_slice.remaining_steps = beam_slice.remaining_steps[sorted_idxes]

    stable_slice = beam_slice.get_subslice(0, stable_count)
    moving_slice = beam_slice.get_subslice(stable_count, beam_slice.size)

    return lost_slice, stable_slice, moving_slice


class PusherAndSolver:
    def __init__(self, config):
        self.config = config
        #self.layout_beam_slice = configure_layout_beam_slice(config)
        self.move_beam_slice = beam_slice_mover(config)
        self.solver = CylindricalPlasmaSolver(config)
        self.r_step = float(config.get('r-step'))
        max_radius = float(config.get('window-width'))
        self.n_cells = int(max_radius / self.r_step) + 1
        self.xi_step_p = config.getfloat('xi-step')
        self.window_length = config.getfloat('window-length')
        self.xi_layers_num = int(self.window_length / self.xi_step_p)

        
    def step_dt(self, plasma_particles, plasma_fields, beam_source, beam_drain, t):
        beam_slice_to_move = BeamSlice(0)
        rho_layout = np.zeros(self.n_cells)
        for layer_idx in np.arange(self.xi_layers_num):
            # Get beam layer with xi \in [xi^{layer_idx + 1}, xi^{layer_idx})
            # Its index is `layer_idx`
            beam_slice_to_layout = beam_source.get_beam_slice(
                (layer_idx - 1) * -self.xi_step_p, layer_idx * -self.xi_step_p,
            )
            rho_beam, rho_layout = layout_beam_slice(
                beam_slice_to_layout,
                layer_idx,
                rho_layout,
                self.r_step,
                self.xi_step_p,
            )
            # Now we can compute plasma layer `layer_idx` reaction
            plasma_particles_new, plasma_fields_new, steps = self.solver.step_dxi(
                plasma_particles, plasma_fields, rho_beam,
            )
            # Now we can move beam layer `layer_idx - 1`
            self.move_beam_slice(
                beam_slice_to_move,
                layer_idx - 1,
                plasma_fields_new,
                plasma_fields,
            )
            
            lost_slice, stable_slice, moving_slice = split_beam_slice(
                beam_slice_to_move, (layer_idx - 1) * -self.xi_step_p,
            )
            beam_drain.push_beam_slice(stable_slice)
            beam_drain.finish_layer((layer_idx - 1) * -self.xi_step_p)
            beam_slice_to_move = beam_slice_to_layout.concatenate(moving_slice)
            plasma_particles = plasma_particles_new
            plasma_fields = plasma_fields_new
            if layer_idx % 100 == 0:
                print('xi={xi:.6f} Ez={Ez:e} N={N}'.format(xi=layer_idx * -self.xi_step_p, Ez=plasma_fields.E_z[0], N=steps))
        return plasma_particles, plasma_fields
