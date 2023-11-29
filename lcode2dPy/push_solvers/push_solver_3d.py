import numpy as np

from ..config.config import Config
from ..plasma3d.data import Arrays
from ..plasma3d.solver import Plane2d3vPlasmaSolver
from ..diagnostics.diagnostics_3d import get
from ..beam3d import BeamSource3D, BeamDrain3D, BeamParticles3D, \
                     BeamCalculator, RigidBeamCalculator


class PusherAndSolver3D:
    def __init__(self, config: Config):
        self.config = config

        self.plasma_solver = Plane2d3vPlasmaSolver(config)
        self.beam_particles_class = BeamParticles3D

        rigid_beam = config.getbool('rigid-beam')
        if rigid_beam:
            self.beam_calculator = RigidBeamCalculator(config)
        else:
            self.beam_calculator = BeamCalculator(config)

        # Import plasma solver and beam pusher, pl = plasma

        self.xi_max = config.getfloat('window-length')
        self.xi_step_size = config.getfloat('xi-step')
        self.xi_steps = round(self.xi_max / self.xi_step_size)
        self.grid_steps = config.getint('window-width-steps')

        # TODO: Get rid of time_step_size and how we change current_time
        #       in step_dt method later, when we figure out how time
        #       in diagnostics should work.
        # self.time_step_size = config.getfloat('time-step')

    def step_dt(self, plasma_fields: Arrays, plasma_particles: Arrays,
                plasma_currents: Arrays, plasma_const_arrays: Arrays,
                xi_plasma_layer_start,
                beam_source: BeamSource3D, beam_drain: BeamDrain3D,
                current_time, diagnostics_list=[]):
        """
        Perform one time step of beam-plasma calculations.
        """
        xp = plasma_const_arrays.xp

        self.beam_calculator.start_time_step()
        beam_layer_to_move = self.beam_particles_class(xp)
        fell_size = 0

        # TODO: Not sure this is right if we start from a saved plasma state and
        #       with a saved beamfile.
        ro_beam_prev = xp.zeros(
            (self.grid_steps, self.grid_steps), dtype=xp.float64)

        xi_i_plasma_layer_start =\
            round(-xi_plasma_layer_start / self.xi_step_size) + 1
        for xi_i in range(xi_i_plasma_layer_start, self.xi_steps + 1, 1):
            beam_layer_to_layout = \
                beam_source.get_beam_layer_to_layout(xi_i)
            ro_beam_full = \
                self.beam_calculator.layout_beam_layer(beam_layer_to_layout,
                                                       xi_i)

            prev_plasma_fields = plasma_fields.copy()

            plasma_particles, plasma_fields, plasma_currents = \
                self.plasma_solver.step_dxi(
                    plasma_particles, plasma_fields, plasma_currents,
                    plasma_const_arrays, ro_beam_full, ro_beam_prev)

            lost, moved, fell_to_next_layer =\
                self.beam_calculator.move_beam_layer(
                    beam_layer_to_move, fell_size, xi_i, prev_plasma_fields,
                    plasma_fields)

            # Creats next beam layer to move and
            # ro_beam_prev for the next iteration:
            beam_layer_to_move, fell_size, ro_beam_prev =\
                self.beam_calculator.create_next_layer(
                    beam_layer_to_layout, fell_to_next_layer, ro_beam_full)

            beam_drain.push_beam_layer(moved)
            # beam_drain.push_beam_lost(lost)

            # Diagnostics:
            xi_plasma_layer = - self.xi_step_size * xi_i

            for diagnostic in diagnostics_list:
                diagnostic.after_step_dxi(
                    current_time, xi_plasma_layer, plasma_particles,
                    plasma_fields, plasma_currents, ro_beam_full)

            # Some diagnostics:            
            if xi_i % 10. == 0:
                Ez_00 = get(
                    plasma_fields.Ez[self.grid_steps//2, self.grid_steps//2])
                print(
                    # f't={current_time:+.4f}, ' + 
                    f'xi={-xi_i * self.xi_step_size:+.4f} Ez={Ez_00:+.4e}')

        # Perform diagnostics
        xi_plasma_layer = - self.xi_step_size * self.xi_steps
        for diagnostic in diagnostics_list:
            diagnostic.dump(current_time, xi_plasma_layer, plasma_particles,
                            plasma_fields, plasma_currents, beam_drain)
