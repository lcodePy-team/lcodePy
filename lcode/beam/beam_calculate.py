import numba as nb
import numpy as np

from .weights import (
    deposit_particles,
    particle_fields,
    beam_particle_weights,
)

from .data import BeamParticles as BeamParticles2D


@nb.vectorize([nb.float64(nb.float64, nb.float64, nb.float64)], cache=True)
def beam_substepping_step(q_m, p_z, substepping_energy):
    """
    Calculate required time step for beam particles.
        
    Paramters 
    ---------
    q_m : array or float64
        'q/m' of the particles.
    p_z : array or float64
        Particles longitudinal momentum.
    substepping_energy : float64
        Critical energy for beam substeping.
    
    Returns
    -------
    dt : array or float64
        'Correct' time step for the considered particles. 
    """
    dt = 1.0
    gamma_mass = np.sqrt(1 / q_m ** 2 + p_z ** 2)
    max_dt = np.sqrt(gamma_mass / substepping_energy)
    while dt > max_dt:
        dt /= 2
    return dt


@nb.njit
def cross_nb(vec1, vec2):
    """
    Vector product.
    """
    result = np.zeros_like(vec1)
    result[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    result[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    result[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    return result


@nb.njit
def in_layer(r_vec, xi_k_1):
    return xi_k_1 <= r_vec[2]


@nb.njit
def is_lost(r_vec, r_max):
    return r_vec[0] ** 2 + r_vec[1] ** 2 >= r_max ** 2


@nb.njit
def beam_to_vec(xi, r, p_z, p_r, M):
    """
    Translate particle parameters from cylindrical to local cartesian 
    coordianates. In the considered position the x-axis is co-directed 
    with the r-axis. Z-axis remains unchanged. The y-coordinate of 
    the particle is 0.
    """
    p_x = p_r
    p_y = M / r
    p_vec = np.array((p_x, p_y, p_z))
    r_vec = np.array((r, 0, xi))
    return p_vec, r_vec


@nb.njit
def vec_to_beam(layer_xi, layer_r, layer_p_z, layer_p_r, 
                layer_M, layer_remaining_steps, layer_id,
                idx, r_vec, p_vec, steps_left, lost, magnetic_field):
    """
    Translate particle parameters from local cartesian coordianates 
    to global cylindrical coordinates. 
    """
    x = r_vec[0]
    y = r_vec[1]
    layer_r[idx] = np.sqrt(x ** 2 + y ** 2)
    layer_xi[idx] = r_vec[2]
    layer_p_r[idx] = (x * p_vec[0] + y * p_vec[1]) / layer_r[idx]
    layer_p_z[idx] = p_vec[2]
    if magnetic_field == 0:
        layer_M[idx] = x * p_vec[1] - y * p_vec[0]
    layer_remaining_steps[idx] = steps_left
    if lost:
        layer_id[idx] = -np.abs(layer_id[idx])


def configure_beam_pusher(config):
    """
    Configurate beam-pusher for axisymmetric case. 

    Parametrs
    ---------
    config : Config
        The set of base parameters to perform the simulation.
    
    Returns
    -------
    push_particles : function
        Function that calculates the motion of the beam particles.
    """
    r_step = float(config.get('window-width-step-size'))
    xi_step_p = float(config.get('xi-step'))
    max_radius = float(config.get('window-width'))
    lost_boundary = max(0.9 * max_radius, max_radius - 1)
    magnetic_field = float(config.get('magnetic-field'))
    locals_spec = {'r_step': nb.float64,
                   'xi_step': nb.float64,
                   'lost_boundary': nb.float64,
                   'magnetic_field': nb.float64,}

    # Moves particles as far as possible on its xi layer
    @nb.njit(locals=locals_spec)
    def push_particles(layer_xi, layer_r, layer_p_z, layer_p_r, layer_M, 
                        layer_q_m, layer_dt, layer_remaining_steps, 
                        layer_lost, layer_size, layer_id, 
                        xi_i, E_r_k_1, E_f_k_1, E_z_k_1,
                        B_f_k_1, B_z_k_1, E_r_k, E_f_k, E_z_k, B_f_k, B_z_k):
        """
        Push beam particles located from xi to xi - dxi as far as possible.  
        
        Parametrs
        ---------
        layer_xi : array
            xi coordinates of the considered beam particles.
        layer_r : array
            r coordinates of the considered beam particles.
        layer_p_z : array
            Longitudinal momentum of the considered beam particles.
        layer_p_r : array
            Transverse momentum of the considered beam particles.
        layer_M : array
            Angular momentum of the considered beam particles.
        layer_q_m : array
            Charge to mass ratio of the considered beam particles.
        layer_dt : array
            Correct" time steps for the considered beam particles 
            taking into account the substepping. 
        layer_remaining_steps : array
            Number of time steps remaining before the beam particles 
            goes to the next time step.
        layer_lost : array
            Boolean array marking the beam particles 
            going outside the simulation box.
        layer_size : int
            Number of beam particles in the considered layer.
        layer_id : array
            Array of unique identifiers of beam particles.
        xi_i : int
            Number of xi steps have been performed by plasma solver. 
        E_r_k_1 : array
            Radial electric field at xi_i step.
        E_f_k_1 : array
            Azimuthal electric field at xi_i step.
        E_z_k_1 : array
            Longitudinal electric field at xi_i step.
        B_f_k_1 : array
            Azimuthal electric field at xi_i step.
        B_z_k_1 : array
            Longitudinal electric field at xi_i step.
        E_r_k : array
            Radial electric field at xi_i - 1 step.
        E_f_k : array
            Azimuthal electric field at xi_i - 1 step.
        E_z_k : array
            Longitudinal electric field at xi_i - 1 step.
        B_f_k : array
            Azimuthal electric field at xi_i - 1 step.
        B_z_k : array
            Longitudinal electric field at xi_i - 1 step.
        
        Returns
        -------
        nlost : int
            The number of particles that have left the simulation box.
        """
        xi_end = xi_i * -xi_step_p
        nlost = 0
        if layer_size == 0:
            return
        for idx in np.arange(layer_size):
            q_m = layer_q_m[idx]
            dt = layer_dt[idx]
            lost = False
            steps = layer_remaining_steps[idx]

            # Initial impulse and position vectors
            p_vec, r_vec = beam_to_vec(layer_xi[idx], layer_r[idx], 
                                              layer_p_z[idx], layer_p_r[idx],
                                              layer_M[idx])
            while steps > 0:
                # Compute approximate position of the particle in the middle of the step
                gamma_mass = np.sqrt((1 / q_m) ** 2 + np.sum(p_vec ** 2))
                r_vec_half_step = r_vec + dt / 2 * p_vec / gamma_mass
                # Add time shift correction (dxi = (v_z - c)*dt)
                r_vec_half_step[2] -= dt / 2

                if not in_layer(r_vec_half_step, xi_end):
                    break
                if is_lost(r_vec_half_step, lost_boundary):
                    # Particle hit the wall and is now lost
                    nlost += 1 
                    layer_lost[idx] = True
                    break

                # Interpolate fields and compute new impulse
                (e_vec, b_vec) = particle_fields(
                    r_vec_half_step,
                    E_r_k_1, E_f_k_1, E_z_k_1, B_f_k_1, B_z_k_1,
                    E_r_k,   E_f_k,   E_z_k,   B_f_k,   B_z_k,
                    xi_end,
                    r_step,
                    xi_step_p,
                )

                p_vec_half_step = p_vec + dt / 2 * np.sign(q_m) * \
                    (e_vec + cross_nb(p_vec / gamma_mass, b_vec))  # Just Lorentz

                # Compute final coordinates and impulses
                gamma_mass = np.sqrt((1 / q_m) ** 2 + np.sum(p_vec_half_step ** 2))
                r_vec += dt * p_vec_half_step / gamma_mass
                # Add time shift correction (dxi = (v_z - c)*dt)
                r_vec[2] -= dt
                p_vec = 2 * p_vec_half_step - p_vec
                steps -= 1

                if is_lost(r_vec, lost_boundary):
                    # Particle hit the wall and is now lost
                    nlost += 1 
                    layer_lost[idx] = True
                    break
            vec_to_beam(layer_xi, layer_r, layer_p_z, layer_p_r, 
                        layer_M, layer_remaining_steps, layer_id,
                        idx, r_vec, p_vec, steps, lost, 
                        magnetic_field)
        return nlost
    return push_particles

@nb.njit
def init_substepping(layer_p_z, layer_q_m, layer_dt, layer_remaining_steps, 
                     time_step, substepping_energy):
    """
    Initialization of substepping for newly received beam particles.
    
    Parametrs
    ---------
    layer_p_z : array
        Longitudinal momentum of the considered beam particles.
    layer_q_m : array
        Charge to mass ratio of the considered beam particles.
    layer_dt : array
        Correct" time steps for the considered beam particles 
        taking into account the substepping. 
        For new particles is 0. 
    layer_remaining_steps : array
        Number of time steps remaining before the beam particles 
        goes to the next time step.
    time_step : float64
        Base time step for current simulation.
    substepping_energy : float64
        Critical energy for beam substeping.
    """
    mask = layer_dt == 0
    dt = beam_substepping_step(layer_q_m[mask], layer_p_z[mask], 
                               substepping_energy)
    steps = (1 / dt).astype(np.int_)
    layer_dt[mask] = dt * time_step
    layer_remaining_steps[mask] = steps


def get_beam_slice_mover(config):
    """
    Configure the beam layer evolution calculator.

    Paramters 
    ---------
    config : Config
        The set of base parameters to perform the simulation.

    Returns
    -------
    move_beam_slice : function
        Function for calculating the evolution of beam particles 
        located between xi and xi - dxi. 
    """
    time_step = float(config.get('time-step'))
    substepping_energy = config.getfloat('beam-substepping-energy')
    push_particles = configure_beam_pusher(config)

    # @nb.njit
    def move_beam_slice(beam_slice, xi_layer_idx, 
                        fields_after_slice, fields_before_slice,):
        """
        Initilize substepping for newly received beam particles.
        After, calculate the evolution of beam particles 
        located between xi and xi - dxi. 

        Paramters 
        ---------
        beam_slice : BeamParticles
            Beam particles to be calculated for the given xi step.
        xi_layer_idx : int
            The step number that was calculated by the plasma solver.
        fields_after_slice : Arrays
            Electric and magnetic fields at xi_layer_idx step.
        fields_before_slice : Arrays
            Electric and magnetic fields at xi_layer_idx - 1 step.
        """
        if beam_slice.size == 0:
            return
        if substepping_energy == 0:
            # Particles with dt != 0 came from the previous time step and
            # need no initialization
            beam_slice.dt[beam_slice.dt == 0] = time_step
        else:
            init_substepping(beam_slice.p_z, beam_slice.q_m, 
                             beam_slice.dt, beam_slice.remaining_steps, 
                             time_step, substepping_energy)

        nlost = push_particles(
                    beam_slice.xi, beam_slice.r,
                    beam_slice.p_z, beam_slice.p_r, beam_slice.M, 
                    beam_slice.q_m, beam_slice.dt, beam_slice.remaining_steps, 
                    beam_slice.lost, beam_slice.size,
                    beam_slice.id, xi_layer_idx, 
                    fields_after_slice.E_r, fields_after_slice.E_f,
                    fields_after_slice.E_z, fields_after_slice.B_f,
                    fields_after_slice.B_z,
                    fields_before_slice.E_r, fields_before_slice.E_f,
                    fields_before_slice.E_z, fields_before_slice.B_f,
                    fields_before_slice.B_z
                )
        beam_slice.nlost += nlost

    return move_beam_slice

class BeamCalculator2D():
    """
    The main class for performing operations with a beam in 2d.  
    """
    def __init__(self, config):
        """
        Initialization.
        
        Paramters 
        ---------
        config : Config
            The set of base parameters to perform the simulation.
        """
        # Get main calculation parameters.
        self.r_step = config.getfloat('window-width-step-size')
        self.grid_steps = int(config.getfloat('window-width') / self.r_step) + 1
        self.xi_step = config.getfloat('xi-step')
        self.move_beam_slice = get_beam_slice_mover(config)

    def start_time_step(self):
        """
        Initialization of array to collect the partial beam density. 
        """
        self.rho_layout = np.zeros(self.grid_steps)

    def layout_beam_layer(self, beam_layer, xi_i):
        """
        Calculate beam density on the grid. 

        Parametrs
        ---------
        beam_layer_to_layout : BeamParticles2D
            Beam particles which should be layout.
        xi_i : int
            The step number to be calculated by the plasma solver next time.

        Returns
        -------
        current_rho_layout : array
            Beam density on the grid at step xi_i. 
        """
        next_rho_layout = np.zeros_like(self.rho_layout)
        xi_end = (xi_i+1) * (-self.xi_step)
        if beam_layer.size != 0:
            j, a00, a01, a10, a11 = beam_particle_weights(beam_layer.r, 
                                                          beam_layer.xi, 
                                                          xi_end, self.r_step, 
                                                          self.xi_step)
            deposit_particles(beam_layer.q_norm, 
                              self.rho_layout, next_rho_layout, 
                              j, a00, a01, a10, a11)

        current_rho_layout = self.rho_layout
        self.rho_layout = next_rho_layout
        current_rho_layout /= self.r_step ** 2
        current_rho_layout[0] *= 6
        current_rho_layout[1:] /= np.arange(1, self.grid_steps)
        return current_rho_layout

    def move_beam_layer(self, beam_partickles_to_move, fell_size, 
                        xi_i, prev_pl_fields, pl_fields):
        """
        Calculate the evolution of beam particles 
        located between xi and xi - dxi.

        Paramters 
        ---------
        beam_partickles_to_move : BeamParticles
            Beam particles to be calculated for the given xi step.
        fell_size : int
            Number of particles coming from the previous step xi.
        xi_i : int
            The step number that was calculated by the plasma solver.
        prev_pl_fields : array
            Electric and magnetic fields at xi_i - 1 step.
        pl_fields : array
            Electric and magnetic fields at xi_i step.

        Returns
        -------
        layers : tuple of arrays
            [lost, move, ramain] beam particles. 
            For more information, see BeamCalculator2D.split_beam_slice. 
        """
        self.move_beam_slice(
            beam_partickles_to_move,
            xi_i,
            pl_fields,
            prev_pl_fields,
        )
        return self.split_beam_slice(beam_partickles_to_move, 
                                     xi_i * -self.xi_step)

    # split beam slice into lost, stable and moving beam slices
    # @nb.njit
    def split_beam_slice(self, beam_slice, xi_end):
        """
            Split beam particles to three groups:
                1. Particles that have been lost from the simulation box. 
                2. Particles that move to the next time step. 
                3. Particles that remain at the current time step.

            Parametrs 
            ---------
            beam_slice : BeamParticles2D
                Beam particles which should be slited. 
            xi_end : float
                Left boundary of the step under consideration.
            
            Returns
            -------
            layers : tuple of arrays
                [lost, move, ramain] beam particles. 
        """
        sorted_idxes = np.argsort(beam_slice.lost)[::-1]                                           
        beam_slice.particles = beam_slice.particles[sorted_idxes]                                  
        beam_slice.dt = beam_slice.dt[sorted_idxes]                                                
        beam_slice.remaining_steps = beam_slice.remaining_steps[sorted_idxes]                      
        beam_slice.lost = beam_slice.lost[sorted_idxes]                                            
        lost_count = np.sum(beam_slice.lost)
        lost_slice = beam_slice.get_subslice(0, lost_count)
        
        beam_slice = beam_slice.get_subslice(lost_count, beam_slice.size)
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

