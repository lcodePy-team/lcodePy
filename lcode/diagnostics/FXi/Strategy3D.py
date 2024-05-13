import numpy as np

from ...config import Config
from ..utils import get

def calculate_energy_fluxes(grid_step_size,
                            plasma_particles, plasma_fields):
    # xp is either np (numpy) or cp (cupy):
    xp = plasma_particles['electrons'].xp

    # The perturbation to the flux density of the electromagnetic energy:
    Se = (plasma_fields.Ez ** 2 + plasma_fields.Bz ** 2 +
         (plasma_fields.Ex - plasma_fields.By) ** 2 +
         (plasma_fields.Ey + plasma_fields.Bx) ** 2) * grid_step_size ** 2 / 2

    # Additional array to calculate the motion energy of plasma particles:
    plasma_particles = plasma_particles['electrons'] # TODO fix it
    gamma_m = xp.sqrt(plasma_particles.m ** 2  + plasma_particles.px ** 2 +
                      plasma_particles.py ** 2 + plasma_particles.pz ** 2)
    motion_energy = gamma_m - plasma_particles.m

    # Calculate integral total energy flux and return it:
    psi_electromagnetic = xp.sum(Se)
    psi_total = xp.sum(motion_energy) + psi_electromagnetic
    return psi_total


class _3D_FXi:
    def __init__(self, config: Config, diagnostic):
        self.__config = config
        self.__diagnostic = diagnostic

        self.__process_probe_lines(diagnostic._probe_lines)

    def __process_probe_lines(self, probe_lines):
        steps = self.__config.getint('window-width-steps')
        grid_step_size = self.__config.getfloat('transverse-step')

        if probe_lines is None:
            self.__ax_x = 0
            self.__ax_y = 0
        else:
            if type(probe_lines) == list or type(probe_lines) == np.ndarray:
                probe_lines = np.array(probe_lines)
            
            if probe_lines.ndim < 1 or probe_lines.ndim > 2:
                raise ValueError('probe_lines must be 1D or 2D')
            
            if probe_lines.ndim == 1:
                self.__ax_x = probe_lines
                self.__ax_y = np.zeros_like(probe_lines)
            else: # ndim == 2
                self.__ax_x = probe_lines[0]
                self.__ax_y = probe_lines[1]

        self.__ax_x = (steps // 2 +
                       np.round(self.__ax_x / grid_step_size)).astype(int)
        self.__ax_y = (steps // 2 +
                       np.round(self.__ax_y / grid_step_size)).astype(int)

    def process_field(self, plasma_fields, field):

        val = getattr(plasma_fields, field)[self.__ax_x, self.__ax_y]
        self.__diagnostic._data[field].append(get(val))
    
    def process_n(self, plasma_currents, n):
        idx = 0 if n == 'ne' else 1 # ni
        val = plasma_currents.ro[idx, self.__ax_x, self.__ax_y]
        self.__diagnostic._data[n].append(get(val))

    def process_rho(self, rho_beam):
        val = rho_beam[self.__ax_x, self.__ax_y]
        self.__diagnostic._data['rho_beam'].append(get(val))
    
    def process_Phi(self, plasma_fields):
        val = plasma_fields.Phi[self.__ax_x, self.__ax_y]
        self.__diagnostic._data['Phi'].append(get(val))

    def process_Sf(self, grid_step_size, plasma_particles, plasma_fields):
        val = calculate_energy_fluxes(grid_step_size,
                                      plasma_particles,
                                      plasma_fields)
        self.__diagnostic._data['Sf'].append(get(val))

    def process_chaotic(self, name, plasma_particles):
        xp = plasma_particles['electrons'].xp
        val = xp.amax(xp.absolute(getattr(plasma_particles['electrons'], f"{name}_chaotic")))
        self.__diagnostic._data[f"{name}_chaotic"].append(get(val))
    
    def process_chaotic_perp(self, name, plasma_particles):
        xp = plasma_particles['electrons'].xp
        kvl_mass = getattr(plasma_particles['electrons'], f"{name}_chaotic")
        a1 = int(xp.shape(kvl_mass)[0] / 6)
        a2 = 5 * a1
        val = xp.amax(xp.absolute(kvl_mass[a1:a2,a1:a2]))
        self.__diagnostic._data[f"{name}_chaotic_perp"].append(get(val))