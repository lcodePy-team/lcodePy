import numpy as np

from ..config.config import Config

from ..beam import BeamSource2D, BeamDrain2D, BeamParticles2D, beam_slice_mover
from ..beam.beam_calculate import layout_beam_slice

from ..plasma.solver import CylindricalPlasmaSolver
from ..plasma.data import Arrays

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
    lost_slice = BeamParticles2D(0)
    # lost_sorted_idxes = np.argsort(beam_slice.lost)
    # beam_slice.particles = beam_slice.particles[lost_sorted_idxes]
    # beam_slice.dt = beam_slice.dt[lost_sorted_idxes]
    # beam_slice.remaining_steps = beam_slice.remaining_steps[lost_sorted_idxes]
    # beam_slice.lost = beam_slice.lost[lost_sorted_idxes]
    # lost_slice = beam_slice.get_subslice(0, beam_slice.nlost)
    # beam_slice = beam_slice.get_subslice(beam_slice.nlost, beam_slice.size)
    

    moving_mask = np.logical_or(beam_slice.remaining_steps > 0, beam_slice.xi < xi_end)
    stable_count = moving_mask.size - np.sum(moving_mask)

    sorted_idxes = np.argsort(moving_mask)
    beam_slice.particles = beam_slice.particles[sorted_idxes]
    beam_slice.dt = beam_slice.dt[sorted_idxes]
    beam_slice.remaining_steps = beam_slice.remaining_steps[sorted_idxes]
    beam_slice.lost = beam_slice.lost[sorted_idxes]
    stable_slice = beam_slice.get_subslice(0, stable_count)
    moving_slice = beam_slice.get_subslice(stable_count, beam_slice.size)

    return lost_slice, stable_slice, moving_slice


class PusherAndSolver2D:
    def __init__(self, config: Config):
        self.config = config
        
        #self.layout_beam_slice = configure_layout_beam_slice(config)
        self.move_beam_slice = beam_slice_mover(config)
        self.solver = CylindricalPlasmaSolver(config)
        self.r_step = float(config.get('window-width-step-size'))
        self.is_rigid = 1 if config.get('rigid-beam')=='y' else 0
        max_radius = float(config.get('window-width'))
        self.n_cells = int(max_radius / self.r_step) + 1
        self.xi_step_p = config.getfloat('xi-step')
        self.window_length = config.getfloat('window-length')
        self.xi_layers_num = int(self.window_length / self.xi_step_p)


    def step_dt(self, plasma_fields: Arrays, plasma_particles: Arrays,
                plasma_currents: Arrays,
                beam_source: BeamSource2D, beam_drain: BeamDrain2D,
                current_time, diagnostics_list=[]):
        beam_slice_to_move = BeamParticles2D(0)
        rho_layout = np.zeros(self.n_cells)
        for layer_idx in np.arange(self.xi_layers_num + 1):
            # Get beam layer with xi \in [xi^{layer_idx + 1}, xi^{layer_idx})
            # Its index is `layer_idx`

            beam_slice_to_layout = beam_source.get_beam_slice(
                layer_idx * -self.xi_step_p, (layer_idx + 1) * -self.xi_step_p,
            )
            rho_beam, rho_layout = layout_beam_slice(
                beam_slice_to_layout,
                layer_idx + 1,
                rho_layout,
                self.r_step,
                self.xi_step_p,
            )

            # Now we can compute plasma layer `layer_idx` reaction
            (plasma_particles, plasma_fields_new, plasma_currents, steps) = \
                self.solver.step_dxi(plasma_particles, plasma_fields,
                                     plasma_currents, rho_beam)

            # Now we can move beam layer `layer_idx - 1`
            self.move_beam_slice(
                beam_slice_to_move,
                layer_idx,
                plasma_fields_new,
                plasma_fields,
            )
            # TODO lost потом обрабатывать
            lost_slice, stable_slice, moving_slice = split_beam_slice(
                beam_slice_to_move, layer_idx * -self.xi_step_p,
            )
            beam_drain.push_beam_slice(stable_slice)
            beam_drain.finish_layer(layer_idx * -self.xi_step_p)
            beam_slice_to_move = beam_slice_to_layout.concatenate(moving_slice)
            plasma_fields = plasma_fields_new
            
            # Every xi step diagnostics
            for diagnostic in diagnostics_list:
                diagnostic.process(
                    self.config, current_time, layer_idx, steps,
                    plasma_particles, plasma_fields, rho_beam, stable_slice)

            # Some diagnostics:
            Ez_00 = plasma_fields_new.E_z[0]

            print(
                f't={current_time:+.4f}, ' + 
                f'xi={-layer_idx * self.xi_step_p:+.4f} Ez={Ez_00:+.4e}'
            )
