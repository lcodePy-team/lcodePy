import numpy as np

from lcode2dPy.config.config import Config

from lcode2dPy.alt_beam_generator.beam_shape import BeamShape

particle_dtype2d = np.dtype([('xi', 'f8'), ('r', 'f8'),
                             ('p_z', 'f8'), ('p_r', 'f8'), ('M', 'f8'),
                             ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'i8')])

particle_dtype3d = np.dtype([('xi', 'f8'), ('x', 'f8'), ('y', 'f8'),
                             ('px', 'f8'), ('py', 'f8'), ('pz', 'f8'),
                             ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'i8')])

def rigid_beam_current(beam_shape: BeamShape, xi_step_p):
    xi_vals = np.arange(-xi_step_p / 2, -beam_shape.total_length, -xi_step_p)
    current = np.zeros_like(xi_vals)
    for idx, xi in np.ndenumerate(xi_vals):
        current[idx] = beam_shape.initial_current(xi)
    return current


def generate_beam(config: Config, beam_shape: BeamShape):
    xi_step_p = config.getfloat('window-width-step-size')
    three_dimensions = (config.get('geometry') == '3d' or
                        config.get('geometry') == '3D')
    max_radius = xi_step_p * config.getint('window-width-steps') / 2
    # Should a user define max_radius by themselves?

    rng = np.random.RandomState(beam_shape.rng_seed)
    current = rigid_beam_current(beam_shape, xi_step_p)
    layers_number = len(current)
    elem_charge = np.abs(2 * beam_shape.current / beam_shape.particles_in_layer)
    particles_in_layers: np.ndarray = (beam_shape.particles_in_layer *
                           np.abs(current / beam_shape.current)).astype(np.int_)
    total_particles = particles_in_layers.sum()
    if three_dimensions:
        beam = np.zeros(total_particles, dtype=particle_dtype3d)
    else:
        beam = np.zeros(total_particles, dtype=particle_dtype2d)
    part_idx = 0
    beam['id'] = np.arange(1, total_particles + 1, dtype=np.int_)

    for layer_idx in np.arange(layers_number):
        layer_xi = -xi_step_p * layer_idx
        xi_middle = layer_xi - xi_step_p / 2
        segment, segment_start = beam_shape.get_segment(xi_middle)
        dxi = segment_start - xi_middle
        particles_in_layer = particles_in_layers[layer_idx]
        # Precalculate common particle properties of this layer
        q_m = (1 / segment.mass_charge_ratio) * (1 if current[layer_idx] > 0 
                                                 else -1)
        q_norm = elem_charge * (1 if beam_shape.current >= 0 else -1)
        start_idx = part_idx
        end_idx = part_idx + particles_in_layer

        beam['xi'][start_idx:end_idx] = np.random.uniform(layer_xi - xi_step_p,
                                                          layer_xi,
                                                          particles_in_layer)
        if three_dimensions:
            x, y, p_x, p_y = segment.get_r_values3d(rng, dxi, max_radius,
                                                    particles_in_layer)
            beam['x'][start_idx:end_idx] = x
            beam['y'][start_idx:end_idx] = y
            beam['px'][start_idx:end_idx] = p_x
            beam['py'][start_idx:end_idx] = p_y
            beam['pz'][start_idx:end_idx] = segment.get_pz(rng, dxi,
                                                        particles_in_layer)
        else:
            r_b, p_br, M_b = segment.get_r_values2d(rng, dxi, max_radius,
                                                    particles_in_layer)
            beam['r'][start_idx:end_idx] = r_b
            beam['p_r'][start_idx:end_idx] = p_br
            beam['M'][start_idx:end_idx] = M_b
            beam['p_z'][start_idx:end_idx] = segment.get_pz(rng, dxi,
                                                        particles_in_layer)
        beam['q_m'][start_idx:end_idx] = np.full(particles_in_layer, q_m)
        beam['q_norm'][start_idx:end_idx] = np.full(particles_in_layer, q_norm)
        part_idx += particles_in_layer
    sort_idxes = np.argsort(-beam['xi'])
    beam = beam[sort_idxes]
    return beam