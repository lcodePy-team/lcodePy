import numpy as np

# def E_z_diag(diagnostics, t, layer_idx, plasma_particles, plasma_fields, rho_beam, beam_slice,
#              t_start, t_end, period, r_selected):
#     time_step = diagnostics.config.getfloat('time-step')
#     if t < t_start or t > t_end:
#         return
#     if t <= t_start + time_step and layer_idx == 0:
#         diagnostics.Ez = {}
#     if (t - t_start) % period != 0:
#         return
#     if layer_idx == 0:
#         diagnostics.Ez[t] = []
#     r_grid_steps = plasma_fields.E_z.size
#     rs = np.linspace(0, diagnostics.config.getfloat('window-width'), r_grid_steps)
#     E_z_selected = plasma_fields.E_z[rs == r_selected][0]
#     diagnostics.Ez[t].append(E_z_selected)
#     return E_z_selected

class Diagnostics:
    def __init__(self, diag_list):
        self.diag_list = diag_list

class TDiagnostics():
    def __init__(self, t_start=0, t_end=None, period=100):
        self.t_start = t_start
        self.t_end = t_end 
        self.period = period  
        self.data = {}
    
    def process(self, config, t, layer_idx, steps, plasma_particles, plasma_fields, rho_beam, beam): 
        if self.t_end and (t<self.t_start or t>self.t_end):
            return False
        elif t<self.t_start:
            return False
        if (t-self.t_start)%self.period != 0:
            return False
        return True

class BeamDiagnostics(TDiagnostics):
    def __init__(self, t_start=0, t_end=None, period=100):
        super().__init__(t_start, t_end, period)
    def process(self, config, t, layer_idx, steps, plasma_particles, plasma_fields, rho_beam, beam):
        if super().process(config, t, layer_idx, steps, plasma_particles, plasma_fields, rho_beam, beam):
            beam_slice = beam
            #lost_slice = beam[1]
            #self.lost[t] = np.array([],dtype=particle_dtype)
            if layer_idx == 0:
                particle_dtype = np.dtype([('xi', 'f8'), ('r', 'f8'), ('p_z', 'f8'), ('p_r', 'f8'), ('M', 'f8'), ('q_m', 'f8'),
                           ('q_norm', 'f8'), ('id', 'i8')])
                self.data[t] = np.array([],dtype=particle_dtype)
            self.data[t] = np.append(self.data[t], beam_slice.particles)
            #self.lost[t] = np.append(self.data[t], lost_slice.particles)
            #self.test[t].append(rho_beam.tolist())