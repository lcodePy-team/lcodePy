import numpy as np
#import openpmd_api as io
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

class MyDiagnostics:
    def __init__(self, config, diag_list):
        self.c = config 
        self.diag_list = diag_list

    def config(self):
        for diag in self.diag_list:
            try:
                diag.config(self.c)
            except AttributeError:
                pass
    
    def process(self,*param):
        for diag in self.diag_list:
            diag.dxi(*param)

    def dump(self):
        for diag in self.diag_list:
            diag.dump()

class FieldDiagnostics:
    def __init__(self, config, name, r=0, 
                 t_start=None, t_end=None, period=100, 
                 cl_mem = False, 
                 out = 'i f',img_format = 'png', is_merge = False,
                 make_gif = False, gif_name=None):
        if name not in ['E_z', 'E_f', 'E_r', 'B_f', 'B_z']:
            raise AttributeError("Name isn't corrected")
        self.name = name
        self.r = r
        self.t_start = t_start
        if t_end is None:
            self.t_end = config.getfloat('time-limit')
        else:
            self.t_end = t_end 
        self.period = period
        self.is_merge = is_merge
        self.out = out
        self.data = {}

    def pull_config(self, config):
        self.dt = config.getfloat('time-step')
        self.tlim = config.getfloat('time-limit')
        if self.t_start is None:
            self.t_start = 0
        if self.t_end is None:
            self.t_end = self.tlim
        r_step = config.getfloat('window-width-step-size')
        xi_step = config.getfloat('xi-step')
        self.w = int(config.getfloat('window-length')/xi_step)+1
        self.h = int(config.getfloat('window-width')/r_step)+1
        
        self.last_idx = self.w - 1  

        
    def process(self, config, t, layer_idx, \
                plasma_particles, plasma_fields, rho_beam, \
                beam_slice):
        if t<self.t_start or t>self.t_end:
            return
        if (t-self.t_start)%self.period == 0:
            if layer_idx == 0:
                if self.r is None:
                    self.data[t] = np.empty((self.w, self.h)) # Grid
                else:
                    self.data[t] = np.empty(self.w)  # Line
            if self.r is None:
                self.data[t][layer_idx] = getattr(plasma_fields, self.name) 
            else:
                self.data[t][layer_idx] = getattr(plasma_fields, self.name)[self.r]
            
            if layer_idx == self.last_idx:
                self.dump(t)
                
    
    def dump(self,t):
        Path('./diagnostics/fields').mkdir(parents=True, exist_ok=True)
        if 'i' in self.out:
            if self.r is None:
                plt.imshow(self.data[t].T)
                plt.savefig(f'./diagnostics/fields/{self.name}_grid_{100*t:08.0f}.png')
                plt.close()
            else:
                plt.plot(self.data[t])
                plt.savefig(f'./diagnostics/fields/{self.name}_{100*t:08.0f}.png')
                plt.close()
        
        if 'f' in self.out:
            if self.r is None:
                np.save(f'./diagnostics/fields/{self.name}_grid_{100*t:08.0f}.npy', self.data)
            else:
                np.save(f'./diagnostics/fields/{self.name}_{100*t:08.0f}.npy',self.data)
    
    # def make_gif(self, name=None):
    #     if name is None:
    #         name = self.name
    #     r_path = './gif_tmp/'
    #     path = Path(r_path)
    #     try:
    #         path.mkdir()
    #     except FileExistsError:
    #         shutil.rmtree(path)
    #         path.mkdir()
    #     files = []
    #     for key in self.data.keys():
    #         if self.r is None:
    #             plt.imshow(self.data[key].T)
    #         else:
    #             plt.ylim(-0.06,0.06) # magic
    #             plt.plot(self.data[key])
                
    #         file = r_path+str(key)+'.png'
    #         files.append(file)
    #         plt.savefig(file)
    #         plt.close()
    #     with imageio.get_writer(name+'.gif', mode='I') as writer:
    #         for filename in files:
    #             image = imageio.imread(filename)
    #             writer.append_data(image)
    #         for i in range(10):
    #             writer.append_data(image)
    #     shutil.rmtree(path)
        
class BeamDiagnostics:
    def __init__(self, config, t_start=0, t_end=None, period=100, 
                 cl_mem = False, 
                 output_type = 'i f', img_format = 'png',
                 make_gif = False, gif_name=None):
        self.t_start = t_start
        if t_end is None:
            self.t_end = config.getfloat('time-limit') 
        else:
            self.t_end = t_end 
        self.period = period 
        self.cl_mem = cl_mem 
        self.output_type = output_type
        self.data = {}
        self.lost = {}

    def pull_config(self, config):
        self.tlim = config.getfloat('time-limit')
        if self.t_end is None:
            self.t_end = self.tlim
        xi_step = config.getfloat('xi-step')
        self.last_idx = int(config.getfloat('window-length')/xi_step) - 1

    def process(self, config, t, layer_idx, \
                plasma_particles, plasma_fields, rho_beam, \
                beam_slice):
        if t<self.t_start or t>self.t_end:
            return
        beam_slice = beam_slice
        #lost_slice = beam[1]
        
            #self.lost[t] = np.array([],dtype=particle_dtype)
        if (t-self.t_start)%self.period == 0:
            if layer_idx == 0:
                particle_dtype = np.dtype([('xi', 'f8'), ('r', 'f8'), ('p_z', 'f8'), ('p_r', 'f8'), ('M', 'f8'), ('q_m', 'f8'),
                           ('q_norm', 'f8'), ('id', 'i8')])
                self.data[t] = np.array([],dtype=particle_dtype)
            self.data[t] = np.append(self.data[t], beam_slice.particles)
            #self.lost[t] = np.append(self.data[t], lost_slice.particles)
            #self.test[t].append(rho_beam.tolist())
            
class PlasmaDiagnostics:
    def __init__(self, config, t_start=0, t_end=None, period=100, 
                 cl_mem = False, 
                 output_type = 'i f', img_format = 'png',
                 make_gif = False, gif_name=None):
        self.t_start = t_start
        if t_end is None:
            self.t_end = config.getfloat('time-limit') 
        else:
            self.t_end = t_end       
        self.period = period 
        self.cl_mem = cl_mem 
        self.output_type = output_type
        self.data = {}
        self.lost = {}

    def pull_config(self, config):
        self.tlim = config.getfloat('time-limit')
        if self.t_end is None:
            self.t_end = self.tlim
        xi_step = config.getfloat('xi-step')
        self.last_idx = int(config.getfloat('window-length')/xi_step) - 1

    def process(self, config, t, layer_idx, \
                plasma_particles, plasma_fields, rho_beam, \
                beam_slice):
        if t<self.t_start or t>self.t_end:
            return
        
        # beam_slice = beam[0]
        # lost_slice = beam[1]
        if layer_idx == 0:
            particle_dtype = np.dtype([('r', 'f8'), ('p_r', 'f8'), ('p_f', 'f8'), ('p_z', 'f8'), ('q', 'f8'), ('m', 'f8'),('age', 'f8')])
            self.data[t] = np.array([],dtype=particle_dtype)
        #     self.lost[t] = np.array([],dtype=particle_dtype)
        if (t-self.t_start)%self.period == 0:
            self.data[t] = np.append(self.data[t], plasma_particles)
            #print(plasma_particles)
            #self.test[t].append(rho_beam.tolist())
            