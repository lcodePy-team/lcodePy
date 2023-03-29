"""The module for experimental noise filters that we are trying to use to successfully pass the so-called test 1."""
from scipy.signal import savgol_filter

from ..config.config import Config
from .data import Arrays


# This function generates a noise filtering function which
# we then use in the predictor-corrector loop (solver.py).
def get_noise_filter(config: Config):
    # Here it is much more convenient to take the config parameters only once,
    # and not just give them to the noise_filter along with the particles arrays.
    xi_step_size = config.getfloat('xi-step')

    filter_window_length = config.getint('filter-window-length')
    filter_polyorder = config.getint('filter-polyorder')
    filter_coefficient = config.getfloat('filter-coefficient')

    # A new noise mitigation method.
    def noise_filter(particles: Arrays, particles_prev: Arrays):
        # First, we take out all the required arrays:
        x_offt, y_offt = particles.x_offt, particles.y_offt
        px, py, pz = particles.px, particles.py, particles.pz

        dx_chaotic_previous = particles_prev.dx_chaotic
        dy_chaotic_previous = particles_prev.dy_chaotic
        dx_chaotic_perp_previous = particles_prev.dx_chaotic_perp
        dy_chaotic_perp_previous = particles_prev.dy_chaotic_perp

        # Particle displacement is (x_offt, y_offt).
        # Longitudinal displacement inhomogeneity:
        dx_inhom = x_offt[1:-1, :] - (x_offt[2:, :] + x_offt[:-2, :]) / 2
        dy_inhom = y_offt[:, 1:-1] - (y_offt[:, 2:] + y_offt[:, :-2]) / 2
        # and transverse (perpendicular):
        dx_inhom_perp = y_offt[1:-1, :] - (y_offt[2:, :] + y_offt[:-2, :]) / 2
        dy_inhom_perp = x_offt[:, 1:-1] - (x_offt[:, 2:] + x_offt[:, :-2]) / 2

        # Sagol filter-averaged longitudinal displacement inhomogeneity:
        dx_inhom_averaged = savgol_filter(
            x=dx_inhom, window_length=filter_window_length,
            polyorder=filter_polyorder, axis=0)
        dy_inhom_averaged = savgol_filter(
            x=dy_inhom, window_length=filter_window_length,
            polyorder=filter_polyorder, axis=1)
        # and transverse (perpendicular):
        dx_inhom_averaged_perp = savgol_filter(
            x=dx_inhom_perp, window_length=filter_window_length,
            polyorder=filter_polyorder, axis=0)
        dy_inhom_averaged_perp = savgol_filter(
            x=dy_inhom_perp, window_length=filter_window_length,
            polyorder=filter_polyorder, axis=1)

        # Chaotic longitudinal displacement:
        dx_chaotic = dx_inhom - dx_inhom_averaged
        dy_chaotic = dy_inhom - dy_inhom_averaged
        # and transverse (perpendicular):
        dx_chaotic_perp = dx_inhom_perp - dx_inhom_averaged_perp
        dy_chaotic_perp = dy_inhom_perp - dy_inhom_averaged_perp

        # Restoring force:
        force_x = - filter_coefficient * dx_chaotic
        force_y = - filter_coefficient * dy_chaotic
        force_x_perp = - filter_coefficient * dx_chaotic_perp
        force_y_perp = - filter_coefficient * dy_chaotic_perp

        # Uncomment this to test a relativism corrected filter:
        # gamma_m = np.sqrt(m**2 + pz**2 + px**2 + py**2)
        # factor_1 = xi_step / (1 - pz / gamma_m)

        # px[1:-1, :] += factor_1[1:-1, :] * force_x
        # px[2:, :]   -= factor_1[2:, :]   * force_x / 2
        # px[:-2, :]  -= factor_1[:-2, :]  * force_x / 2

        # py[:, 1:-1] += factor_1[1:-1, :] * force_y
        # py[:, 2:]   -= factor_1[2:, :]   * force_y / 2
        # py[:, :-2]  -= factor_1[:-2, :]  * force_y / 2

        # A filter without relativism corrections:
        px[1:-1, :] += xi_step_size * force_x
        px[2:, :]   -= xi_step_size * force_x / 2
        px[:-2, :]  -= xi_step_size * force_x / 2
        py[1:-1, :] += xi_step_size * force_x_perp
        py[2:, :]   -= xi_step_size * force_x_perp / 2
        py[:-2, :]  -= xi_step_size * force_x_perp / 2

        py[:, 1:-1] += xi_step_size * force_y
        py[:, 2:]   -= xi_step_size * force_y / 2
        py[:, :-2]  -= xi_step_size * force_y / 2
        px[:, 1:-1] += xi_step_size * force_y_perp
        px[:, 2:]   -= xi_step_size * force_y_perp / 2
        px[:, :-2]  -= xi_step_size * force_y_perp / 2

        return Arrays(particles.xp,q=particles.q, m=particles.m,
                      x_init=particles.x_init, y_init=particles.y_init,

                      # And arrays that change after filtering:
                      x_offt=x_offt, y_offt=y_offt,
                      px=px, py=py, pz=pz,
                      dx_chaotic=dx_chaotic, dx_chaotic_perp=dx_chaotic_perp,
                      dy_chaotic=dy_chaotic, dy_chaotic_perp=dy_chaotic_perp)

    return noise_filter