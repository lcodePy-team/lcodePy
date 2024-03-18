"""The module for experimental noise filters that we are trying to use to successfully pass the so-called test 1."""
import numpy as np
from ..config.config import Config
from .data import Arrays


# This function generates a noise filtering function which
# we then use in the predictor-corrector loop (solver.py).
def get_noise_filter(config: Config):
    # Here it is much more convenient to take the config parameters only once,
    # and not just give them to the noise_filter along with the particles arrays.
    xi_step_size = config.getfloat('xi-step')

    filter_window_length = config.getint('declustering-averaging')
    filter_coefficient = config.getfloat('declustering-force')
    damping_coefficient = config.getfloat('damping-declustering')
    dx_max = config.getfloat('declustering-limit')

    pu_type = config.get('processing-unit-type').lower()
    if pu_type == 'cpu':
        from scipy.ndimage import fourier_gaussian
    elif pu_type == 'gpu':
        from cupyx.scipy.ndimage import fourier_gaussian

    def calculate_inhomogeneity(d_offt, xp: np, axis):
        if axis: d_offt = d_offt.T
        d_inhom = xp.vstack((
            d_offt[0, :] - 2 * d_offt[1, :] + d_offt[2, :],
            d_offt[1:-1, :] - (d_offt[2:, :] + d_offt[:-2, :]) / 2,
            d_offt[-1, :] - 2 * d_offt[-2, :] + d_offt[-3, :]))
        if axis: d_inhom = d_inhom.T
        return d_inhom

    # A new noise mitigation method.
    def noise_filter(particles: Arrays, particles_prev: Arrays):
        # Zero step. We determine if we will use numpy or cupy as xp:
        xp = particles.xp

        # First, we take out all the required arrays:
        x_offt, y_offt = particles.x_offt, particles.y_offt
        px, py, pz = particles.px, particles.py, particles.pz

        dx_chaotic_previous = particles_prev.dx_chaotic
        dy_chaotic_previous = particles_prev.dy_chaotic
        # dx_chaotic_perp_previous = particles_prev.dx_chaotic_perp
        # dy_chaotic_perp_previous = particles_prev.dy_chaotic_perp

#         # Particle displacement is (x_offt, y_offt).
#         # Longitudinal displacement inhomogeneity:
#         dx_inhom = x_offt[1:-1, :] - (x_offt[2:, :] + x_offt[:-2, :]) / 2
#         dy_inhom = y_offt[:, 1:-1] - (y_offt[:, 2:] + y_offt[:, :-2]) / 2
#         # and transverse (perpendicular):
#         dx_inhom_perp = y_offt[1:-1, :] - (y_offt[2:, :] + y_offt[:-2, :]) / 2
#         dy_inhom_perp = x_offt[:, 1:-1] - (x_offt[:, 2:] + x_offt[:, :-2]) / 2
        
        dx_inhom = (calculate_inhomogeneity(x_offt, xp, 0) + calculate_inhomogeneity(x_offt, xp, 1)) / 2
        dy_inhom = (calculate_inhomogeneity(y_offt, xp, 1) + calculate_inhomogeneity(y_offt, xp, 0)) / 2
        # Зануление смещений на границе, пока только ухудшало все
        # dx_inhom[0,:] *= 0
        # dx_inhom[-1,:] *= 0
        # dx_inhom[:,0] *= 0
        # dx_inhom[:,-1] *= 0
        # dy_inhom[0,:] *= 0
        # dy_inhom[-1,:] *= 0
        # dy_inhom[:,0] *= 0
        # dy_inhom[:,-1] *= 0

        # dx_inhom = calculate_inhomogeneity(x_offt, xp, 0)
        # dy_inhom = calculate_inhomogeneity(y_offt, xp, 1)
        # dx_inhom_perp = calculate_inhomogeneity(y_offt, xp, 0)
        # dy_inhom_perp = calculate_inhomogeneity(x_offt, xp, 1)

        dx_inhom_averaged = xp.fft.ifft2( fourier_gaussian(xp.fft.fft2(dx_inhom), sigma=filter_window_length) ).real  
        dy_inhom_averaged = xp.fft.ifft2( fourier_gaussian(xp.fft.fft2(dy_inhom), sigma=filter_window_length) ).real  
        # Sagol filter-averaged longitudinal displacement inhomogeneity:
        # dx_inhom_averaged = savgol_filter(
        #     x=dx_inhom, window_length=filter_window_length,
        #     polyorder=filter_polyorder, axis=0)
        # dx_inhom_averaged = savgol_filter(
        #     x=dx_inhom_averaged, window_length=filter_window_length,
        #     polyorder=filter_polyorder, axis=1)
        # dy_inhom_averaged = savgol_filter(
        #     x=dy_inhom, window_length=filter_window_length,
        #     polyorder=filter_polyorder, axis=0)
        # dy_inhom_averaged = savgol_filter(
        #     x=dy_inhom_averaged, window_length=filter_window_length,
        #     polyorder=filter_polyorder, axis=1)
        # and transverse (perpendicular):
        # dx_inhom_averaged_perp = savgol_filter(
        #     x=dx_inhom_perp, window_length=filter_window_length,
        #     polyorder=filter_polyorder, axis=0)
        # dy_inhom_averaged_perp = savgol_filter(
        #     x=dy_inhom_perp, window_length=filter_window_length,
        #     polyorder=filter_polyorder, axis=1)

        # Chaotic longitudinal displacement:
        dx_chaotic = dx_inhom - dx_inhom_averaged
        dy_chaotic = dy_inhom - dy_inhom_averaged
        # and transverse (perpendicular):
        # dx_chaotic_perp = dx_inhom_perp - dx_inhom_averaged_perp
        # dy_chaotic_perp = dy_inhom_perp - dy_inhom_averaged_perp

        # d_limit = (dx_chaotic * dx_chaotic +
        #            dy_chaotic * dy_chaotic +
        #            dx_chaotic_perp * dx_chaotic_perp +
        #            dy_chaotic_perp * dy_chaotic_perp)
        d_limit = (dx_chaotic * dx_chaotic +
                   dy_chaotic * dy_chaotic)
        d_limit = xp.where(d_limit < dx_max**2, 1 - d_limit/dx_max**2, 0)
        
        # Restoring force:
        force_x = - filter_coefficient * dx_chaotic * d_limit
        force_y = - filter_coefficient * dy_chaotic * d_limit
        # force_x_perp = - filter_coefficient * dx_chaotic_perp * d_limit
        # force_y_perp = - filter_coefficient * dy_chaotic_perp * d_limit

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
        # px[1:-1, :] += xi_step_size * (force_x  - damping_coefficient * d_limit *
        #                                (dx_chaotic - dx_chaotic_previous))[1:-1, :]
        # px[2:, :]   -= xi_step_size * force_x[1:-1, :] / 2
        # px[:-2, :]  -= xi_step_size * force_x[1:-1, :] / 2
        # py[1:-1, :] += xi_step_size * (force_x_perp - damping_coefficient * d_limit *
        #                                (dx_chaotic_perp - dx_chaotic_perp_previous))[1:-1, :]
        # py[2:, :]   -= xi_step_size * force_x_perp[1:-1, :] / 2
        # py[:-2, :]  -= xi_step_size * force_x_perp[1:-1, :] / 2
        # py[:, 1:-1] += xi_step_size * (force_y - damping_coefficient * d_limit *
        #                                (dy_chaotic - dy_chaotic_previous))[:, 1:-1]
        # py[:, 2:]   -= xi_step_size * force_y[:, 1:-1] / 2
        # py[:, :-2]  -= xi_step_size * force_y[:, 1:-1] / 2
        # px[:, 1:-1] += xi_step_size * (force_y_perp - damping_coefficient * d_limit *
        #                                (dy_chaotic_perp - dy_chaotic_perp_previous))[:, 1:-1]
        # px[:, 2:]   -= xi_step_size * force_y_perp[:, 1:-1] / 2
        # px[:, :-2]  -= xi_step_size * force_y_perp[:, 1:-1] / 2

        px[1:-1, :] += xi_step_size * force_x[1:-1, :] / 2
        px[2:, :]   -= xi_step_size * force_x[1:-1, :] / 4
        px[:-2, :]  -= xi_step_size * force_x[1:-1, :] / 4
        px[:, 1:-1] += xi_step_size * force_x[:, 1:-1] / 2
        px[:, 2:]   -= xi_step_size * force_x[:, 1:-1] / 4
        px[:, :-2]  -= xi_step_size * force_x[:, 1:-1] / 4
        px -= xi_step_size * damping_coefficient * d_limit * (dx_chaotic - dx_chaotic_previous)

        py[1:-1, :] += xi_step_size * force_y[1:-1, :] / 2
        py[2:, :]   -= xi_step_size * force_y[1:-1, :] / 4
        py[:-2, :]  -= xi_step_size * force_y[1:-1, :] / 4
        py[:, 1:-1] += xi_step_size * force_y[:, 1:-1] / 2
        py[:, 2:]   -= xi_step_size * force_y[:, 1:-1] / 4
        py[:, :-2]  -= xi_step_size * force_y[:, 1:-1] / 4
        py -= xi_step_size * damping_coefficient * d_limit * (dy_chaotic - dy_chaotic_previous)
        return Arrays(particles.xp, q=particles.q, m=particles.m,
                      x_init=particles.x_init, y_init=particles.y_init,

                      # And arrays that change after filtering:
                      x_offt=x_offt, y_offt=y_offt,
                      px=px, py=py, pz=pz,
                      # dx_chaotic=dx_chaotic, dx_chaotic_perp=dx_chaotic_perp,
                      # dy_chaotic=dy_chaotic, dy_chaotic_perp=dy_chaotic_perp)
                      dx_chaotic=dx_chaotic, dx_chaotic_perp=particles_prev.dx_chaotic_perp,
                      dy_chaotic=dy_chaotic, dy_chaotic_perp=particles_prev.dy_chaotic_perp)

    return noise_filter