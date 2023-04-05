import numpy as np

from ..config.config import Config

#import for 2D simulation
from ..beam import BeamParticles2D, BeamCalculator2D
from ..plasma.solver import CylindricalPlasmaSolver

#import for 3D simulation
from ..plasma3d.data import Arrays, ArraysView
from ..plasma3d.solver import Plane2d3vPlasmaSolver
from ..beam3d import BeamParticles3D
from ..beam3d import BeamCalculator as BeamCalculator3D

class PusherAndSolver():
    """
    Parent class for calculation xi-cycle. 
    """
    def __init__(self, config: Config):
        """
        Initializes the inner state according config.

        Paramters 
        ---------
        config : Config
            The set of base parameters to perform the simulation.
        """
        self.config = config
        
        #remeber some config value
        self.dxi = config.getfloat('xi-step')
        self.grid_steps = config.getint('window-width-steps') 
        if not self.grid_steps:
            max_radius = config.getfloat('window-width')
            r_step = config.getfloat('window-width-step-size')
            self.grid_steps = int(max_radius / r_step) + 1
        window_length = config.getfloat('window-length')
        self.xi_steps = int(window_length / self.dxi)
    
    def _set_beam_particles(self, xp):
        pass
    def _set_rho_beam_array(self, xp, grid_steps):
        pass
    def _get_beam_layer(self, beam_source, xi_i):
        pass
    def _simple_diag(self, current_time, xi_i, pl_fields):
        pass

    def step_dt(self, pl_fields, pl_particles,
                pl_currents, pl_const_arrays,
                beam_source, beam_drain,
                current_time, diagnostics_list=[]):
        """
        Perform one time step of beam-plasma calculations.
        
        NOTE: The data structure is different for 2D and 3D so far. 
            The parameter type is written as `2D-data (3D-data)`.

        Parameters
        ----------
        pl_fileds : Array
                The collection of Er, Ef, Ez, Bf, Bz (and Br for 3D). 
        
        pl_particles : Array
                All plasma partcles.  
        
        pl_currents : Array
                The collection of plasma macroparameters j and rho. 
        
        pl_cons_arrays : Array
                Pre-calculated coefficients and initial state for plasma solver.
        
        beam_source : BeamSource2D (BeamSource3D)
                The source of beam particles. 
                It provides the particles for a given time step.
        
        beam_source : BeamDrain2D (BeamDrain3D)
                The drain of beam particles.   
                It provides particle transfer to the next time step.
        """
        xp = pl_const_arrays.xp

        self.beam_calc.start_time_step()
        beam_layer_to_move = self._set_beam_particles(xp)
        fell_size = 0

        # TODO: Not sure this is right if we start from a saved plasma state and
        #       with a saved beamfile.
        #       Do we need array here?
        rho_beam_prev = self._set_rho_beam_array(xp, self.grid_steps)

        for xi_i in np.arange(self.xi_steps + 1):

            # Get beam particles with xi in [dxi*{xi_i + 1}, dxi*{xi_i})
            # This use to finish rho_beam[xi_i]
            beam_layer_to_layout = self._get_beam_layer(beam_source, xi_i)

            rho_beam = self.beam_calc.layout_beam_layer(beam_layer_to_layout,
                                                        xi_i)

            # Save fields from xi_i - 1 step for beam moving 
            prev_pl_fields = pl_fields.copy()

            # Now we can compute plasma layer `xi_i` reaction
            pl_particles, pl_fields, pl_currents = \
                self.solver.step_dxi(pl_particles, pl_fields,
                                     pl_currents, pl_const_arrays,
                                     rho_beam, rho_beam_prev)

            # Now we can move beam with xi in [dxi*{xi_i - 1}, dxi*{xi_i})
            lost, moved, fell_to_next_layer = \
                self.beam_calc.move_beam_layer(beam_layer_to_move, 
                                               fell_size, xi_i, prev_pl_fields, 
                                               pl_fields)

            rho_beam_prev = rho_beam.copy()

            # Add pircticles with xi in [dxi*{xi_i + 1}, dxi*{xi_i}) 
            # for move it in next xi step
            beam_layer_to_move = \
                beam_layer_to_layout.append(fell_to_next_layer)
            fell_size = fell_to_next_layer.id.size

            # Send moved beam particles to next time step 
            beam_drain.push_beam_slice(moved)
            # beam_drain.finish_layer(xi_i * -self.dxi)
            
            # Every xi step diagnostics
            for diagnostic in diagnostics_list:
                diagnostic.process(
                    self.config, current_time, xi_i, 
                    pl_particles, pl_fields, rho_beam, moved)

            self._simple_diag(current_time, xi_i, pl_fields)


class PusherAndSolver2D(PusherAndSolver):
    """
    Class for calculation xi-cycle in 2D axisymmetric geometry. 
    """
    def __init__(self, config: Config):
        """
        Initializes the correct set of computational functions.

        Paramters 
        ---------
        config : Config
            The set of base parameters to perform the simulation.
        """
        super().__init__(config)
        
        self.solver = CylindricalPlasmaSolver(config)
        self.beam_calc = BeamCalculator2D(config)
    
    def _set_beam_particles(self, xp):
        return BeamParticles2D(0)

    def _set_rho_beam_array(self, xp, grid_steps):
        return xp.zeros(grid_steps, dtype=xp.float64)
    
    def _get_beam_layer(self, beam_source, xi_i):
        return beam_source.get_beam_slice(
            xi_i * -self.dxi, (xi_i + 1) * -self.dxi,
        )
    
    def _simple_diag(self, current_time, xi_i, pl_fields):
            # Some diagnostics:
            Ez_00 = pl_fields.E_z[0]

            print(
                f't={current_time:+.4f}, ' + 
                f'xi={-xi_i * self.dxi:+.4f} Ez={Ez_00:+.16e}'
            )

class PusherAndSolver3D(PusherAndSolver):
    """
    Class for calculation xi-cycle in 3D geometry. 
    """
    def __init__(self, config: Config):
        """
        Initializes the correct set of computational functions.

        Paramters 
        ---------
        config : Config
            The set of base parameters to perform the simulation.
        """
        super().__init__(config)

        self.solver = Plane2d3vPlasmaSolver(config)
        self.beam_particles_class = BeamParticles3D
        self.beam_calc = BeamCalculator3D(config)

        # TODO: Get rid of time_step_size and how we change current_time
        #       in step_dt method later, when we figure out how time
        #       in diagnostics should work.
        # self.time_step_size = config.getfloat('time-step')

    def _set_beam_particles(self, xp):
        return BeamParticles3D(xp)

    def _set_rho_beam_array(self, xp, grid_steps):
        return xp.zeros((grid_steps, grid_steps), dtype=xp.float64)
    
    def _get_beam_layer(self, beam_source, xi_i):
        return beam_source.get_beam_layer_to_layout(xi_i)

    def _simple_diag(self, current_time, xi_i, pl_fields):
        view_pl_fields = ArraysView(pl_fields)
        Ez_00 = view_pl_fields.Ez[self.grid_steps//2, self.grid_steps//2]

        print(
            f't={current_time:+.4f}, ' +
            f'xi={-xi_i * self.dxi:+.4f} Ez={Ez_00:+.16e}'
        )