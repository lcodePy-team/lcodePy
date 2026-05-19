import os
import glob
import numpy as np

from lcode.simulation import Simulation
from lcode.diagnostics import ParticlesDiag
from lcode.diagnostics import SliceDiag, SliceType, SliceValue

def analyze_beams(name):
    """
    Post simulation witness processing.
    """
    beam = np.load('diagnostics/' + name + '/beam_00005.00.npz')
    # Select only witness particles by their ID.
    idmin = min(beam['id'][beam['xi'] < -13])
    
    files = sorted(glob.glob('diagnostics/' + name + '/beam*npz')) 
    ts, xs, ys, sx, sy, emx, emy, uzs = [], [], [], [], [], [], [], []
    for file in files:
        beam = np.load(file)
        time = float(file[-12:-4])
        x = beam['x'][beam['id']>=idmin]
        y = beam['y'][beam['id']>=idmin] 
        ux = beam['ux'][beam['id']>=idmin]
        uy = beam['uy'][beam['id']>=idmin]
        uz = beam['uz'][beam['id']>=idmin]
        xav = np.mean(x)
        x2av = np.mean(x * x)
        uxav = np.mean(ux)
        yav = np.mean(y)
        y2av = np.mean(y * y)
        uyav = np.mean(uy)
        ts.append(time)
        xs.append(xav)
        ys.append(yav)
        sx.append(np.sqrt(x2av - xav**2))
        sy.append(np.sqrt(y2av - yav**2))
        emx.append(np.sqrt((x2av-xav**2) * (np.mean(ux*ux) - uxav**2) 
                            - (np.mean(x*ux) - xav*uxav)**2
                            ))
        emy.append(np.sqrt((y2av-yav**2) * (np.mean(uy*uy) - uyav**2) 
                            - (np.mean(y*uy) - yav*uyav)**2
                            ))
        uzs.append(np.mean(uz))
    
    np.savez(f'res-{name}.npz', t=np.array(ts), 
             x=np.array(xs), sx=np.array(sx), emx=np.array(emx), 
             y=np.array(ys), sy=np.array(sy), emy=np.array(emy), 
             uz=np.array(uzs))

def get_diagnostics_list(name):
    """
    Setting the same diagnostics for different variants.
    """
    return [SliceDiag(
                SliceType.XI_X,
                output_period=100, directory_name = name, 
                slice_value = SliceValue.ne),
            ParticlesDiag(
                save_beam=True, directory_name = name,
                output_period=10)
           ]

# Set necessary parameters of the configs.
config = {
    'geometry': '3d',
    'processing-unit-type': 'gpu',
    'window-width': 4,
    'transverse-step': 0.02,
    'window-length': 10,
    'xi-step': 0.02,
    'time-limit': 0.01,
    'time-step': 0.01,
    
    'plasma-particles-per-cell': 9,
    'ion-model' : 'background',
    'save-plasma-each-time': 0.005
}

# Obtain the initial state of plasma with an excited wakefield.
beam_parameters = {
    'current': -0.2, 
    'particles_in_layer': 20000,
    'beam1': {'ampl': 1, 'xishape':'c', 'length': 7.1, 'radius': 1, 
              'energy': 1e10, 'angspread': 1e-30},
}

sim = Simulation(config=config, beam_parameters=beam_parameters)
sim.step()


# Update config for long simulations.
config['plasma-shape'] = 'from-file'
config['path-to-plasma-state'] = './plasma_states/'
config['window-length'] = 14.3
config['time-limit'] = 2000 
config['time-step'] = 5
config['save-plasma-each-time'] = 0

# Offset along the x axis.
xshift = 0.4

# Misalignment injection without damping.
name = f'no-damping-{xshift}'
if (not os.path.exists(f'res-{name}.npz')):
    # Set beam parametrs.
    beam_parameters = {
        'current': -0.0268, 
        'particles_in_layer': 200,
        'gap': {'ampl': 0, 'length': 13.66},
        'witness': {'ampl': 1, 'xishape':'T', 'length': 0.52, 'radius': 0.03, 
                    'xshift': xshift, 'energy': 150 / 0.511, 
                    'angspread': 0.00124}
    }
    
    # Get diagnostics configuration. 
    diags = get_diagnostics_list(name)

    # Init new simulation. 
    sim = Simulation(config=config, 
                     diagnostics = diags,
                     beam_parameters=beam_parameters)
    
    # Simulation.
    sim.step()

    # Postprocessing.
    analyze_beams(name)

# Offset along the x axis.
xshifts = [0.2, 0.4]

# Misalignment injection with damping.
for xshift in xshifts:
    name = f'damping-{xshift}'
    if (not os.path.exists(f'res-{name}.npz')):
        
        lh = 0.19
        lc = 0.37
        gap3 = 3.66 - lc - lh

        # Set beam paramters.
        beam_parameters = {
            'current': -0.0268, 
            'particles_in_layer': 200,
            'gap1': {'ampl': 0, 'length': 10},
            'gap2': {'ampl': 0, 'length': lc-lh},
            'damper': {'ampl': 1, 'xishape':'l', 'length': 2*lh, 
                       'radius': 0.03, 'xshift': xshift, 'energy': 150 / 0.511, 
                       'angspread': 0.00124},
            'gap3': {'ampl': 0, 'length': gap3},
            'witness': {'ampl': 1, 'xishape':'T', 'length': 0.52, 
                        'radius': 0.03, 'xshift': xshift, 
                        'energy': 150 / 0.511, 'angspread': 0.00124}
        }
        
        # Get diagnostics configuration. 
        diags = get_diagnostics_list(name)

        # Init new simulation. 
        sim = Simulation(config=config, 
                         diagnostics = diags,
                         beam_parameters=beam_parameters)
        
        # Simulation.
        sim.step()

        # Postprocessing.
        analyze_beams(name)
