"""The module for experimental noise filters that we are trying to use to successfully pass the so-called test 1."""
from scipy.signal import savgol_filter

from ..config.config import Config
from .data import Arrays


def get_noise_filter(config: Config):
    xi_step_size     = config.getfloat('xi-step')

    filter_window_length = config.getint('filter-window-length')
    filter_polyorder = config.getint('filter-polyorder')
    filter_coefficient = config.getfloat('filter-coefficient')

    def noise_filter(particles: Arrays):
        # A new noise mitigation method. Firstly, we get all get all required arrays:
        x_init, y_init = particles.x_init, particles.y_init
        x_offt, y_offt = particles.x_offt, particles.y_offt
        px, py, pz = particles.px, particles.py, particles.pz

        # Particle displacement is (x_offt, y_offt). Displacement inhomogeneity:
        dx_inhom = x_offt[1:-1, :] - (x_offt[2:, :] + x_offt[:-2, :]) / 2
        dy_inhom = y_offt[:, 1:-1] - (y_offt[:, 2:] + y_offt[:, :-2]) / 2

        # Sagol filter-averaged displacement inhomogeneity:
        dx_inhom_averaged = savgol_filter(
            x=dx_inhom, window_length=filter_window_length,
            polyorder=filter_polyorder, axis=0)
        dy_inhom_averaged = savgol_filter(
            x=dy_inhom, window_length=filter_window_length,
            polyorder=filter_polyorder, axis=1)

        # Chaotic displacement:
        dx_chaotic = dx_inhom - dx_inhom_averaged
        dy_chaotic = dy_inhom - dy_inhom_averaged

        # Restoring force:
        force_x = - filter_coefficient * dx_chaotic
        force_y = - filter_coefficient * dy_chaotic

        # Uncomment this to test a relativism corrected filter:
        # gamma_m = np.sqrt(m**2 + pz**2 + px**2 + py**2)
        # factor_1 = xi_step / (1 - pz / gamma_m)

        # px[1:-1, :] += factor_1[1:-1, :] * force_x
        # px[2:, :]   -= factor_1[2:, :]   * force_x / 2
        # px[:-2, :]  -= factor_1[:-2, :]  * force_x / 2

        # py[:, 1:-1] += factor_1[1:-1, :] * force_y
        # py[:, 2:]   -= factor_1[2:, :]   * force_y / 2
        # py[:, :-2]  -= factor_1[:-2, :]  * force_y / 2

        # A filter with relativism corrections:
        px[1:-1, :] += xi_step_size * force_x
        px[2:, :]   -= xi_step_size * force_x / 2
        px[:-2, :]  -= xi_step_size * force_x / 2

        py[:, 1:-1] += xi_step_size * force_y
        py[:, 2:]   -= xi_step_size * force_y / 2
        py[:, :-2]  -= xi_step_size * force_y / 2

        return Arrays(particles.xp, x_init=x_init, y_init=y_init,
                      x_offt=x_offt, y_offt=y_offt,
                      px=px, py=py, pz=pz, q=particles.q, m=particles.m)

    return noise_filter