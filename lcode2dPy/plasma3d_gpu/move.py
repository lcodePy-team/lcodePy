"""Module for calculation of particles movement and interpolation of fields on particles positions."""
from ..config.config import Config
from .weights import weight_cupy
from .data import GPUArrays


# Field interpolation and particle movement (fused) #

# TODO: Not very smart to get a kernel this way.
def get_move_smart_kernel_cupy():
    import cupy as cp

    return cp.ElementwiseKernel(
        in_params="""
        float64 xi_step_size, float64 reflect_boundary,
        float64 grid_step_size, float64 grid_steps,
        raw T m, raw T q, raw T x_init, raw T y_init,
        raw T prev_x_offt, raw T prev_y_offt,
        raw T estim_x_offt, raw T estim_y_offt,
        raw T prev_px, raw T prev_py, raw T prev_pz,
        raw T Ex_avg, raw T Ey_avg, raw T Ez_avg,
        raw T Bx_avg, raw T By_avg, raw T Bz_avg
        """,
        out_params="""
        raw T out_x_offt, raw T out_y_offt,
        raw T out_px, raw T out_py, raw T out_pz
        """,
        operation="""
        const T x_halfstep = x_init[i] + (prev_x_offt[i] + estim_x_offt[i]) / 2;
        const T y_halfstep = y_init[i] + (prev_y_offt[i] + estim_y_offt[i]) / 2;
        
        const T x_h = x_halfstep / (T) grid_step_size + 0.5;
        const T y_h = y_halfstep / (T) grid_step_size + 0.5;
        const T x_loc = x_h - floor(x_h) - 0.5;
        const T y_loc = y_h - floor(y_h) - 0.5;
        const int ix = floor(x_h) + floor(grid_steps / 2);
        const int iy = floor(y_h) + floor(grid_steps / 2);

        T Ex = 0, Ey = 0, Ez = 0, Bx = 0, By = 0, Bz = 0;
        for (int kx = -2; kx <= 2; kx++) {
            const double wx = weight(x_loc, kx);
            for (int ky = -2; ky <= 2; ky++) {
                const double w = wx * weight(y_loc, ky);
                const int idx = (iy + ky) + (int) grid_steps * (ix + kx);

                Ex += Ex_avg[idx] * w; Bx += Bx_avg[idx] * w;
                Ey += Ey_avg[idx] * w; By += By_avg[idx] * w;
                Ez += Ez_avg[idx] * w; Bz += Bz_avg[idx] * w;
            }
        }

        T px = prev_px[i], py = prev_py[i], pz = prev_pz[i];
        const T opx = prev_px[i], opy = prev_py[i], opz = prev_pz[i];
        T x_offt = prev_x_offt[i], y_offt = prev_y_offt[i];

        T gamma_m = sqrt(m[i]*m[i] + px*px + py*py + pz*pz);
        T vx = px / gamma_m, vy = py / gamma_m, vz = pz / gamma_m;
        T factor = q[i] * (T) xi_step_size / (1 - pz / gamma_m);
        T dpx = factor * (Ex + vy * Bz - vz * By);
        T dpy = factor * (Ey - vx * Bz + vz * Bx);
        T dpz = factor * (Ez + vx * By - vy * Bx);
        px = opx + dpx / 2; py = opy + dpy / 2; pz = opz + dpz / 2;

        gamma_m = sqrt(m[i]*m[i] + px*px + py*py + pz*pz);
        vx = px / gamma_m, vy = py / gamma_m, vz = pz / gamma_m;
        factor = q[i] * (T) xi_step_size / (1 - pz / gamma_m);
        dpx = factor * (Ex + vy * Bz - vz * By);
        dpy = factor * (Ey - vx * Bz + vz * Bx);
        dpz = factor * (Ez + vx * By - vy * Bx);
        px = opx + dpx / 2; py = opy + dpy / 2; pz = opz + dpz / 2;

        gamma_m = sqrt(m[i]*m[i] + px*px + py*py + pz*pz);
        x_offt += px / (gamma_m - pz) * xi_step_size;
        y_offt += py / (gamma_m - pz) * xi_step_size;
        px = opx + dpx; py = opy + dpy; pz = opz + dpz;

        T x = x_init[i] + x_offt, y = y_init[i] + y_offt;
        if (x > reflect_boundary) {
            x =  2 * reflect_boundary  - x;
            x_offt = x - x_init[i];
            px = -px;
        }
        if (x < -reflect_boundary) {
            x = -2 * reflect_boundary - x;
            x_offt = x - x_init[i];
            px = -px;
        }
        if (y > reflect_boundary) {
            y = 2 * reflect_boundary  - y;
            y_offt = y - y_init[i];
            py = -py;
        }
        if (y < -reflect_boundary) {
            y = -2 * reflect_boundary - y;
            y_offt = y - y_init[i];
            py = -py;
        }

        out_x_offt[i] = x_offt; out_y_offt[i] = y_offt;
        out_px[i] = px; out_py[i] = py; out_pz[i] = pz;

        """,
        name='move_smart_cupy', preamble=weight_cupy
    )


def get_move_wo_fields_kernel_cupy():
    import cupy as cp

    return cp.ElementwiseKernel(
        in_params="""
        float64 xi_step_size, float64 reflect_boundary,
        raw T m, raw T q, raw T x_init, raw T y_init,
        raw T prev_x_offt, raw T prev_y_offt,
        raw T prev_px, raw T prev_py, raw T prev_pz
        """,
        out_params="""
        raw T out_x_offt, raw T out_y_offt,
        raw T out_px, raw T out_py, raw T out_pz
        """,
        operation="""
        T x_offt = prev_x_offt[i], y_offt = prev_y_offt[i];
        T px = prev_px[i], py = prev_py[i], pz = prev_pz[i];
        const T gamma_m = sqrt(m[i]*m[i] + px*px + py*py + pz*pz);

        x_offt += px / (gamma_m - pz) * xi_step_size;
        y_offt += py / (gamma_m - pz) * xi_step_size;

        T x = x_init[i] + x_offt, y = y_init[i] + y_offt;
        if (x > reflect_boundary) {
            x =  2 * reflect_boundary  - x;
            x_offt = x - x_init[i];
            px = -px;
        }
        if (x < -reflect_boundary) {
            x = -2 * reflect_boundary - x;
            x_offt = x - x_init[i];
            px = -px;
        }
        if (y > reflect_boundary) {
            y = 2 * reflect_boundary  - y;
            y_offt = y - y_init[i];
            py = -py;
        }
        if (y < -reflect_boundary) {
            y = -2 * reflect_boundary - y;
            y_offt = y - y_init[i];
            py = -py;
        }

        out_x_offt[i] = x_offt; out_y_offt[i] = y_offt;
        out_px[i] = px; out_py[i] = py; out_pz[i] = pz;

        """,
        name='move_wo_fields_cupy'
    )


def get_move_smart_func(xi_step_size, reflect_boundary, grid_step_size,
                        grid_steps):
    import cupy as cp

    move_smart_kernel_cupy = get_move_smart_kernel_cupy()

    def move_smart(fields, particles, estimated_particles):
        """
        Update plasma particle coordinates and momenta according to the field
        values interpolated halfway between the previous plasma particle location
        and the the best estimation of its next location currently available to us.
        This is a convenience wrapper around the `move_smart_kernel` CUDA kernel.
        """
        x_offt_new = cp.zeros_like(particles.x_offt)
        y_offt_new = cp.zeros_like(particles.y_offt)
        px_new = cp.zeros_like(particles.px)
        py_new = cp.zeros_like(particles.py)
        pz_new = cp.zeros_like(particles.pz)

        x_offt_new, y_offt_new, px_new, py_new, pz_new = move_smart_kernel_cupy(
            xi_step_size, reflect_boundary, grid_step_size, grid_steps,
            particles.m, particles.q, particles.x_init, particles.y_init,
            particles.x_offt, particles.y_offt,
            estimated_particles.x_offt,
            estimated_particles.y_offt,
            particles.px, particles.py, particles.pz,
            fields.Ex, fields.Ey, fields.Ez,
            fields.Bx, fields.By, fields.Bz,
            x_offt_new, y_offt_new, px_new, py_new, pz_new,
            size=(particles.m).size
        )

        return GPUArrays(x_init=particles.x_init, y_init=particles.y_init,
                        x_offt=x_offt_new, y_offt=y_offt_new,
                        px=px_new, py=py_new, pz=pz_new, 
                        q=particles.q, m=particles.m)
    
    move_wo_fields_kernel_cupy = get_move_wo_fields_kernel_cupy()

    def move_wo_fields(particles):
        """
        Move coarse plasma particles as if there were no fields.
        Also reflect the particles from `+-reflect_boundary`.
        """
        x_offt_new = cp.zeros_like(particles.x_offt)
        y_offt_new = cp.zeros_like(particles.y_offt)
        px_new = cp.zeros_like(particles.px)
        py_new = cp.zeros_like(particles.py)
        pz_new = cp.zeros_like(particles.pz)

        x_offt_new, y_offt_new, px_new, py_new, pz_new =\
            move_wo_fields_kernel_cupy(
                xi_step_size, reflect_boundary, particles.m, particles.q,
                particles.x_init, particles.y_init,
                particles.x_offt, particles.y_offt,
                particles.px, particles.py, particles.pz,
                x_offt_new, y_offt_new, px_new, py_new, pz_new,
                size=(particles.m).size)

        return GPUArrays(x_init=particles.x_init, y_init=particles.y_init,
                        x_offt=x_offt_new, y_offt=y_offt_new,
                        px=px_new, py=py_new, pz=pz_new,
                        q=particles.q, m=particles.m)

    return move_smart, move_wo_fields


class ParticleMover:
    def __init__(self, config: Config):
        xi_step_size          = config.getfloat('xi-step')
        reflect_padding_steps = config.getint('reflect-padding-steps')
        grid_step_size   = config.getfloat('window-width-step-size')
        grid_steps       = config.getint('window-width-steps')
        reflect_boundary = grid_step_size * (
            grid_steps / 2 - reflect_padding_steps)

        self.move_particles, self.move_particles_wo_fields =\
            get_move_smart_func(xi_step_size, reflect_boundary, grid_step_size,
                                grid_steps)
