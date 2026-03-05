import numpy as np
import math
import os

from ..config.config import Config

#import for 2D simulation
from ..beam import BeamParticles2D, BeamCalculator2D
from ..plasma.solver import CylindricalPlasmaSolver

#import for 3D simulation
from ..plasma3d.solver import Plane2d3vPlasmaSolver
from ..beam3d import BeamParticles3D, BeamCalculator, RigidBeamCalculator
from ..diagnostics.diagnostics_3d import get

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
            r_step = config.getfloat('transverse-step')
            self.grid_steps = int(max_radius / r_step) + 1
        window_length = config.getfloat('window-length')
        self.xi_steps = int(window_length / self.dxi)

        self.save_plasma_each_time = config.getfloat('save-plasma-each-time')
        if self.save_plasma_each_time:
            self.time_step = config.getfloat('time-step')
            os.makedirs('./plasma_states', exist_ok=True)

        #Final plasma state for tests
        self._plasmastate = None
    
    
    def _set_beam_particles(self, xp):
        pass
    def _set_rho_beam_array(self, xp, grid_steps):
        pass
    def _get_beam_layer(self, beam_source, xi_i):
        pass
    def _push_beam_layer(self, beam_drain, moved):
        pass
    def _simple_diag(self, current_time, xi_i, pl_fields):
        pass
    def _save_plasma_state(self, current_time, xi, 
                           particles, fields, currents, const_arrays):
        pass

    def step_dt(self, pl_fields, pl_particles,
                pl_currents, pl_const_arrays, xi_plasma_layer_start, 
                beam_source, beam_drain,
                current_time, diagnostics_list=[]):
        """
        Perform one time step of beam-plasma calculations.
        
        NOTE: The data structure is different for 2D and 3D so far. 
            The parameter type is written as `2D-data (3D-data)`.

        Parameters
        ----------
        pl_fileds : Array
                The collection of Er, Ef, Ez, Bf, Bz (for 2D) 
                or Ex, Ey, Ez, Bx, By, Bz (for 3D). 
        
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
        
        current_time : float64
                Time step to be calculated.
        
        diagnostic_list : List
                List of of requested diagnostics. 
        """
        xp = pl_const_arrays.xp

        self.beam_calc.start_time_step()
        beam_layer_to_move = self._set_beam_particles(xp)
        fell_size = 0

        rho_beam_prev = self._set_rho_beam_array(xp, self.grid_steps)
        xi_i_plasma_layer_start = round(-xi_plasma_layer_start / self.dxi) + 1
        xi_plasma_layer = xi_i_plasma_layer_start * self.dxi
        for xi_i in range(xi_i_plasma_layer_start, self.xi_steps + 1, 1):
            # Get beam particles with xi in [dxi*{xi_i + 1}, dxi*{xi_i})
            # This use to finish rho_beam[xi_i]
            beam_layer_to_layout = self._get_beam_layer(beam_source, xi_i)

            rho_beam = self.beam_calc.layout_beam_layer(beam_layer_to_layout,
                                                        xi_i)

            # Save fields from xi_i - 1 step for beam pusher 
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
            self._push_beam_layer(beam_drain, moved) 
            
            xi_plasma_layer = -xi_i * self.dxi
            # Every xi step diagnostics
            for diagnostic in diagnostics_list:
                diagnostic.after_step_dxi(
                    current_time, xi_plasma_layer, pl_particles,
                    pl_fields, pl_currents, rho_beam)
            if xi_i % 10 == 0:
                self._simple_diag(current_time, xi_i, pl_fields)

        for diagnostic in diagnostics_list:
            diagnostic.dump(current_time, xi_plasma_layer, pl_particles,
                            pl_fields, pl_currents, beam_drain)

        if self.save_plasma_each_time:
            self._save_plasma_state(current_time, xi_plasma_layer, 
                                    pl_particles, pl_fields, pl_currents,
                                    pl_const_arrays) 
        self._plasmastate = (pl_particles, pl_fields, pl_currents)


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
        self.pp_dtype = np.dtype([('q', 'f8'), ('m', 'f8'), ('r', 'f8'), 
                                  ('p_r', 'f8'), ('p_f', 'f8'), ('p_z', 'f8'),
                                  ('age', 'f8')
                                  ])
        self.pp_attrs = ('q', 'm', 'r', 'p_r', 'p_f', 'p_z', 'age')
        self.field_components = ('E_r', 'E_f', 'E_z', 'B_f', 'B_z')
        self.currents_components = ('rho', 'j_r', 'j_f', 'j_z')
    
    def _set_beam_particles(self, xp):
        return BeamParticles2D(xp)

    def _set_rho_beam_array(self, xp, grid_steps):
        return xp.zeros(grid_steps, dtype=xp.float64)
    
    def _get_beam_layer(self, beam_source, xi_i):
        return beam_source.get_beam_slice(
            xi_i * -self.dxi, (xi_i + 1) * -self.dxi,
        )
    
    def _push_beam_layer(self, beam_drain, moved):
        beam_drain.push_beam_slice(moved)
    
    def _simple_diag(self, current_time, xi_i, pl_fields):
            # Some diagnostics:
            Ez_00 = pl_fields.E_z[0]

            print(
                f't={current_time:+.4f}, ' + 
                f'xi={-xi_i * self.dxi:+.4f} Ez={Ez_00:+.16e}', flush=True
            )

    def _save_plasma_state(self, current_time, xi, 
                           particles, fields, currents, const_arrays):
        if (abs(math.remainder(current_time, self.save_plasma_each_time)) 
            <= self.time_step / 2):
            particles_to_file = {}
            for sort in const_arrays.sorts:
                data = np.zeros(particles[sort].q.shape, dtype=self.pp_dtype)
                for attr in self.pp_attrs:
                    data[attr] = getattr(particles[sort], attr)
                particles_to_file[sort] = data
            fields_to_file = {}
            for f_comp in self.field_components:
                fields_to_file[f_comp] = getattr(fields, f_comp)
            currents_to_file = {}
            for c_comp in self.currents_components:
                currents_to_file[c_comp] = getattr(currents, c_comp)
            if 'ions' in const_arrays.sorts:
                ni = {}
            else:
                ni = {'ni': const_arrays.ni}
            np.savez(f'./plasma_states/{current_time:09.2f}.npz',
                     **particles_to_file,
                     **fields_to_file,
                     **currents_to_file,
                     **ni,
                     xi_plasma_layer = xi)




class PusherAndSolver3D(PusherAndSolver):
    """
    Class for calculation xi-cycle in 3D cartesian geometry. 
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
        rigid_beam = config.getbool('rigid-beam')
        if rigid_beam:
            self.beam_calc = RigidBeamCalculator(config)
        else:
            self.beam_calc = BeamCalculator(config)
        self.pp_dtype = np.dtype([('q', 'f8'), ('m', 'f8'),
                                  ('x_init', 'f8'), ('y_init', 'f8'),
                                  ('x_offt', 'f8'), ('y_offt', 'f8'),
                                  ('px', 'f8'), ('py', 'f8'), ('pz', 'f8'),
                                  ('dx_chaotic', 'f8'), ('dy_chaotic', 'f8'),
                                  ('dx_chaotic_perp', 'f8'), 
                                  ('dy_chaotic_perp', 'f8'),
                                ])
        self.pp_attrs = ('q', 'm', 'x_init', 'y_init', 'x_offt', 'y_offt', 
                         'px', 'py', 'pz', 'dx_chaotic', 'dy_chaotic')
        self.field_components = ('Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'Phi')
        self.currents_components = ('ro', 'jx', 'jy', 'jz')
        
        self.config = config
    
    def _set_beam_particles(self, xp):
        return BeamParticles3D(xp)

    def _set_rho_beam_array(self, xp, grid_steps):
        return xp.zeros((grid_steps, grid_steps), dtype=xp.float64)
    
    def _get_beam_layer(self, beam_source, xi_i):
        return beam_source.get_beam_layer_to_layout(xi_i)
    
    def _push_beam_layer(self, beam_drain, moved):
        beam_drain.push_beam_layer(moved)

    
    def _simple_diag(self, current_time, xi_i, pl_fields):
            # Some diagnostics:
            Ez_00 = get(pl_fields.Ez[self.grid_steps//2, self.grid_steps//2])
            print(f't={current_time:+.4f}, ' + 
                  f'xi={-xi_i * self.dxi:+.4f} Ez={Ez_00:+.4e}', 
                  flush=True)

    def _save_plasma_state(self, current_time, xi, 
                           particles, fields, currents, const_arrays):
        if (abs(math.remainder(current_time, self.save_plasma_each_time)) 
            <= self.time_step / 2):
            particles_to_file = {}
            for sort in const_arrays.sorts:
                data = np.zeros(particles[sort].q.shape, dtype=self.pp_dtype)
                for attr in self.pp_attrs:
                    data[attr] = get(getattr(particles[sort], attr))
                particles_to_file[sort] = data
            fields_to_file = {}
            for f_comp in self.field_components:
                fields_to_file[f_comp] = get(getattr(fields, f_comp))
            currents_to_file = {}
            for c_comp in self.currents_components:
                currents_to_file[c_comp] = get(getattr(currents, c_comp))
            if 'ions' in const_arrays.sorts:
                ni = {}
            else:
                ni = {'rho_initial': const_arrays.ro_initial}
            np.savez(f'./plasma_states/{current_time:09.2f}.npz',
                     **particles_to_file,
                     **fields_to_file,
                     **currents_to_file,
                     **ni,
                     xi_plasma_layer = xi)