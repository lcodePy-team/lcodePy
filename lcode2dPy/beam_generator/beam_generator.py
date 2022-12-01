import scipy.stats as stats
import os
import numpy as np
from numpy import sqrt, pi
from ..config.config import find 
from ..config.default_config import default_config 
from .beam_profiles import distrs_from_shapes, get_segments_from_c_config, find_beam_profile_pars, split_into_segments

particle_dtype2d = np.dtype([('xi', 'f8'), ('r', 'f8'),
                             ('p_z', 'f8'), ('p_r', 'f8'), ('M', 'f8'),
                             ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'i8')])

particle_dtype3d = np.dtype([('xi', 'f8'), ('x', 'f8'), ('y', 'f8'),
                             ('px', 'f8'), ('py', 'f8'), ('pz', 'f8'),
                             ('q_m', 'f8'), ('q_norm', 'f8'), ('id', 'i8')])

geom_to_particle_dtype = {'c': particle_dtype2d, 
                          'circ': particle_dtype2d,
                          '3d': particle_dtype3d}
geom_to_distr_list = {
    'circ':['xi', 'r', 'p_z', 'p_r', 'M'],
    'c':   ['xi', 'r', 'p_z', 'p_r', 'M'],
    'p':   ['xi', 'x', 'p_x', 'p_z', 'M'],
    '3d':  ['xi', 'x', 'y', 'px', 'py', 'pz'],
}

def make_beam(config, distrs, q_m=1.0, partic_in_layer=200, identifier=0,
              savehead=False, saveto=False, name='beamfile.bin'):
        if saveto and name in os.listdir(saveto):
            raise Exception(
            """Another beamfile with the same name is found.
            You may delete it using the following command:
            "rm %s".""" % os.path.join(saveto, name)
            )
        
        geom = config.get('geometry')
        distr_list = geom_to_distr_list[geom]
        particle_dtype  = geom_to_particle_dtype[geom]
        
        ##### xi-distribution generation ######
        xi_distr = distrs[distr_list[0]]
        xi_step = config.getfloat('xi-step')
        q = 2. * xi_distr.amp / partic_in_layer
        xi = xi_distr(partic_in_layer, xi_step)
        
        if savehead: # beam cut
            cond = xi >= -config.getfloat('window-length')
        else:
            cond = (xi >= -config.getfloat('window-length')) & (xi <= 0)
        xi = xi[cond]
        partic_num = xi.size    
        xi = np.sort(xi)[::-1]
        vals = {'xi': xi}
        ##### other distributions generation ######
        for distr_name in distr_list[1:]:
            vals[distr_name] = distrs[distr_name](partic_num)
            np.random.shuffle(vals[distr_name])
        
        if geom == 'c' or geom == 'circ':
            vals['M'] = vals['M'] * vals['r']
        
        ##### beam construction ######
        vals['q_m'] = q_m * np.ones(partic_num)
        vals['q_norm'] = q * np.ones(partic_num)
        vals['id'] = np.arange(partic_num, dtype=int)
        particles = np.array(list(vals.values()))
        stub_particle = np.zeros(len(particles))
        stub_particle[0] = -100000.
        stub_particle[-3] = 1.
        stub_particle = np.array([stub_particle])
        particles = np.vstack([particles.T, stub_particle])
        particles = np.array(list(map(tuple, particles)),
                             dtype=particle_dtype)
        
        ##### saving data ######
        beam = particles[particles['xi'] <= 0]
        if saveto:
            beam.tofile(os.path.join(saveto, name))
        if savehead:
            head = particles[particles['xi'] > 0]
            head.tofile(os.path.join(saveto, 'head-' + name))
        return particles

def make_beam_from_c_beam_profile(config, beam_profile, beam_current,
 partic_in_layer=200, savehead=False, saveto=False, name='beamfile.bin'):
    beam_profile_parsed = find_beam_profile_pars(beam_profile)
    segments = split_into_segments(beam_profile_parsed)
    
    ##### beam generation ######
    beam = None
    for i, segment in enumerate(segments):
        new_beam = make_beam(config, 
            distrs_from_shapes(segment, beam_current),
              q_m=1/segment['m/q'], identifier=i,
                partic_in_layer=partic_in_layer)
        if beam is None:
            beam = new_beam
        else:
            beam = np.hstack([beam[:-1], new_beam])
    beam[::-1].sort(order='xi')
    
    ##### saving data ######
    if savehead:
        head = beam[beam['xi'] > 0]
        head.tofile(os.path.join(saveto, 'head-' + name))
    beam = beam[beam['xi'] <= 0]
    if saveto:
        beam.tofile(os.path.join(saveto, name))
    return beam


def make_beam_from_c_config(config, path, partic_in_layer=None, savehead=False, saveto=False, name='beamfile.bin'):
    config.update_from_c_config(path)
    beam_profile = config.get_c_beam_profile(path)
    beam_current = config.get_float_from_c_config(path, 'beam-current')
    if partic_in_layer is None:
        partic_in_layer = config.get_float_from_c_config(path, 'beam-particles-in-layer')
    
    beam = make_beam_from_c_beam_profile(config, beam_profile, beam_current, partic_in_layer=partic_in_layer, savehead=savehead, saveto=saveto, name=name)

    return beam