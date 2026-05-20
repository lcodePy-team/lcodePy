from ..beam3d.weights import weight1_cupy, weight4_cupy
# Higuera-Cary pusher. 
# https://doi.org/10.1063/1.4979989
update_momentum_higuera_cary = """
    // First half electric field acceleration
    T qm_dt_2 = q_m * dt / 2.;
    T ux_m = ux + qm_dt_2*Ex_loc;
    T uy_m = uy + qm_dt_2*Ey_loc;
    T uz_m = uz + qm_dt_2*Ez_loc;

    // Calculate the gamma for rotation by a magnetic field:
    T gamma_m_sq = 1. + ux_m*ux_m + uy_m*uy_m + uz_m*uz_m;

    T bx = qm_dt_2 * Bx_loc;
    T by = qm_dt_2 * By_loc;
    T bz = qm_dt_2 * Bz_loc;
    T b_sq = bx*bx + by*by + bz*bz;
    T sigma_2 = (gamma_m_sq - b_sq) / 2;
    T b_dot_u = bx*ux_m + by*uy_m + bz*uz_m;
    T gamma_new = sqrt(sigma_2
                       + sqrt(sigma_2*sigma_2 + b_sq + b_dot_u*b_dot_u));

    // Calculate auxiliary values:
    T tx = bx / gamma_new;
    T ty = by / gamma_new;
    T tz = bz / gamma_new;
    T t_sq_pl = 2 / (1. + tx*tx + ty*ty + tz*tz);
    T sx = tx * t_sq_pl;
    T sy = ty * t_sq_pl;
    T sz = tz * t_sq_pl;

    // Rotation step + second half electric field acceleration: 
    T ux_prime = ux_m + uy_m*tz - uz_m*ty;
    T uy_prime = uy_m + uz_m*tx - ux_m*tz;
    T uz_prime = uz_m + ux_m*ty - uy_m*tx;
    ux = ux_m + uy_prime*sz - uz_prime*sy + qm_dt_2*Ex_loc;
    uy = uy_m + uz_prime*sx - ux_prime*sz + qm_dt_2*Ey_loc;
    uz = uz_m + ux_prime*sy - uy_prime*sx + qm_dt_2*Ez_loc;
"""

# 4-order Runge-Kutta method. 
update_momentum_runge_kutta = """
    // First half electric field acceleration
    T qm_dt = q_m * dt;
    Ex_loc *= qm_dt;
    Ey_loc *= qm_dt;
    Ez_loc *= qm_dt;
    Bx_loc *= qm_dt;
    By_loc *= qm_dt;
    Bz_loc *= qm_dt;

    T vx = ux / gamma;
    T vy = uy / gamma;
    T vz = uz / gamma;
    T k1_x = Ex_loc + vy*Bz_loc - vz*By_loc;
    T k1_y = Ey_loc + vz*Bx_loc - vx*Bz_loc;
    T k1_z = Ez_loc + vx*By_loc - vy*Bx_loc;

    T ux_half = ux + k1_x/2;
    T uy_half = uy + k1_y/2;
    T uz_half = uz + k1_z/2;
    T gamma_half = sqrt(1 
        + ux_half*ux_half + uy_half*uy_half + uz_half*uz_half);
    T vx_half = ux_half / gamma_half;
    T vy_half = uy_half / gamma_half;
    T vz_half = uz_half / gamma_half;
    T k2_x = Ex_loc + vy_half*Bz_loc - vz_half*By_loc;
    T k2_y = Ey_loc + vz_half*Bx_loc - vx_half*Bz_loc;
    T k2_z = Ez_loc + vx_half*By_loc - vy_half*Bx_loc;

    ux_half = ux + k2_x/2;
    uy_half = uy + k2_y/2;
    uz_half = uz + k2_z/2;
    gamma_half = sqrt(1 + ux_half*ux_half + uy_half*uy_half + uz_half*uz_half);
    vx_half = ux_half / gamma_half;
    vy_half = uy_half / gamma_half;
    vz_half = uz_half / gamma_half;
    T k3_x = Ex_loc + vy_half*Bz_loc - vz_half*By_loc;
    T k3_y = Ey_loc + vz_half*Bx_loc - vx_half*Bz_loc;
    T k3_z = Ez_loc + vx_half*By_loc - vy_half*Bx_loc;
    
    ux_half = ux + k3_x;
    uy_half = uy + k3_y;
    uz_half = uz + k3_z;
    gamma_half = sqrt(1 + ux_half*ux_half + uy_half*uy_half + uz_half*uz_half);
    vx_half = ux_half / gamma_half;
    vy_half = uy_half / gamma_half;
    vz_half = uz_half / gamma_half;
    T k4_x = Ex_loc + vy_half*Bz_loc - vz_half*By_loc;
    T k4_y = Ey_loc + vz_half*Bx_loc - vx_half*Bz_loc;
    T k4_z = Ez_loc + vx_half*By_loc - vy_half*Bx_loc;

    ux += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6;
    uy += (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6;
    uz += (k1_z + 2*k2_z + 2*k3_z + k4_z) / 6;
"""

# Boris pusher: J.P. Boris, Relativistic plasma simulation-optimization 
# of a hybrid code, in: Proceedings of 4th Conference on Numerical Simulation 
# of Plasmas, Naval  Research Laboratory, Washington D.C., 1970, pp. 3–67
update_momentum_boris = """
    // First half electric field acceleration
    T qm_dt_2 = q_m * dt / 2;
    T ux_m = ux + qm_dt_2*Ex_loc;
    T uy_m = uy + qm_dt_2*Ey_loc;
    T uz_m = uz + qm_dt_2*Ez_loc;

    // Calculate the gamma for rotation by a magnetic field:
    T gamma_m = sqrt(1. + ux_m*ux_m + uy_m*uy_m + uz_m*uz_m);

    // Calculate auxiliary values:
    T _t = qm_dt_2 / gamma_m;
    T tx = _t * Bx_loc;
    T ty = _t * By_loc;
    T tz = _t * Bz_loc;
    T t_sq_pl = 2 / (1. + tx*tx + ty*ty + tz*tz);
    T sx = tx * t_sq_pl;
    T sy = ty * t_sq_pl;
    T sz = tz * t_sq_pl;

    // Rotation step + second half electric field acceleration: 
    T ux_prime = ux_m + uy_m*tz - uz_m*ty;
    T uy_prime = uy_m + uz_m*tx - ux_m*tz;
    T uz_prime = uz_m + ux_m*ty - uy_m*tx;
    ux = ux_m + uy_prime*sz - uz_prime*sy + qm_dt_2*Ex_loc;
    uy = uy_m + uz_prime*sx - ux_prime*sz + qm_dt_2*Ey_loc;
    uz = uz_m + ux_prime*sy - uy_prime*sx + qm_dt_2*Ez_loc;
"""


# Boris pusher: J.P. Boris, Relativistic plasma simulation-optimization 
# of a hybrid code, in: Proceedings of 4th Conference on Numerical Simulation 
# of Plasmas, Naval  Research Laboratory, Washington D.C., 1970, pp. 3–67
update_momentum_boris_tg = """
    // First half electric field acceleration
    T qm_dt_2 = q_m * dt / 2;
    T ux_m = ux + qm_dt_2*Ex_loc;
    T uy_m = uy + qm_dt_2*Ey_loc;
    T uz_m = uz + qm_dt_2*Ez_loc;

    // Calculate the gamma for rotation by a magnetic field:
    T gamma_m = sqrt(1. + ux_m*ux_m + uy_m*uy_m + uz_m*uz_m);

    // Calculate auxiliary values:
    T B_abs = sqrt(Bx_loc*Bx_loc + By_loc*By_loc + Bz_loc*Bz_loc) + 1e-16;
    T _t = tan(qm_dt_2 * B_abs / gamma_m) / B_abs;
    T tx = _t * Bx_loc;
    T ty = _t * By_loc;
    T tz = _t * Bz_loc;
    T t_sq_pl = 2 / (1. + tx*tx + ty*ty + tz*tz);
    T sx = tx * t_sq_pl;
    T sy = ty * t_sq_pl;
    T sz = tz * t_sq_pl;

    // Rotation step + second half electric field acceleration: 
    T ux_prime = ux_m + uy_m*tz - uz_m*ty;
    T uy_prime = uy_m + uz_m*tx - ux_m*tz;
    T uz_prime = uz_m + ux_m*ty - uy_m*tx;
    ux = ux_m + uy_prime*sz - uz_prime*sy + qm_dt_2*Ex_loc;
    uy = uy_m + uz_prime*sx - ux_prime*sz + qm_dt_2*Ey_loc;
    uz = uz_m + ux_prime*sy - uy_prime*sx + qm_dt_2*Ez_loc;
"""

# Exact geration pusher by Seiji Zenitani and Takayuki Umeda.
# https://doi.org/10.1063/1.5051077
update_momentum_exact_gyration = """
    // First half electric field acceleration:
    T qm_dt_2 = q_m * dt / 2;
    T ux_m = ux + qm_dt_2*Ex_loc;
    T uy_m = uy + qm_dt_2*Ey_loc;
    T uz_m = uz + qm_dt_2*Ez_loc;

    // Calculate the gamma for rotation by a magnetic field:
    T gamma_m = sqrt(1. + ux_m*ux_m + uy_m*uy_m + uz_m*uz_m);

    // Calculate auxiliary values:
    T B_abs = sqrt(Bx_loc*Bx_loc + By_loc*By_loc + Bz_loc*Bz_loc) + 1e-16;
    T Bx_unit = Bx_loc / B_abs;
    T By_unit = By_loc / B_abs;
    T Bz_unit = Bz_loc / B_abs;
    T B_unit_dot_p_m = Bx_unit*ux_m + By_unit*uy_m + Bz_unit*uz_m;
    T ux_m_par = B_unit_dot_p_m * Bx_unit;
    T uy_m_par = B_unit_dot_p_m * By_unit;
    T uz_m_par = B_unit_dot_p_m * Bz_unit;
    T teta = q_m * dt * B_abs / gamma_m;
    T sin_val = sin(teta);
    T cos_val = cos(teta);

    // Rotation step + second half electric field acceleration: 
    ux = (ux_m_par + (ux_m - ux_m_par) * cos_val
          + (uy_m*Bz_unit - uz_m*By_unit) * sin_val
          + qm_dt_2 * Ex_loc);
    uy = (uy_m_par + (uy_m - uy_m_par) * cos_val
          + (uz_m*Bx_unit - ux_m*Bz_unit) * sin_val
          + qm_dt_2 * Ey_loc);
    uz = (uz_m_par + (uz_m - uz_m_par) * cos_val
          + (ux_m*By_unit - uy_m*Bx_unit) * sin_val
          + qm_dt_2 * Ez_loc);
"""


# VD1 pusher by K.V. Vshivkov, E.S. Voropaeva, A.A. Efimova
# https://doi.org/10.25743/ICT.2023.282.004
update_momentum_vd1 = """
    T qm_dt = q_m * dt;
    
    // Calculate parallel abd perpendicular values:
    T B_abs = sqrt(Bx_loc*Bx_loc + By_loc*By_loc + Bz_loc*Bz_loc) + 1e-16;
    T Bx_unit = Bx_loc / B_abs;
    T By_unit = By_loc / B_abs;
    T Bz_unit = Bz_loc / B_abs;
    T B_unit_dot_u = Bx_unit*ux + By_unit*uy + Bz_unit*uz;
    T B_unit_dot_E = Bx_unit*Ex_loc + By_unit*Ey_loc + Bz_unit*Ez_loc;
    T Ex_loc_par = B_unit_dot_E * Bx_unit;
    T Ey_loc_par = B_unit_dot_E * By_unit;
    T Ez_loc_par = B_unit_dot_E * Bz_unit;
    T Ex_loc_perp = Ex_loc - Ex_loc_par;
    T Ey_loc_perp = Ey_loc - Ey_loc_par;
    T Ez_loc_perp = Ez_loc - Ez_loc_par;
    T ux_par = B_unit_dot_u * Bx_unit;
    T uy_par = B_unit_dot_u * By_unit;
    T uz_par = B_unit_dot_u * Bz_unit;
    T ux_perp = ux - ux_par;
    T uy_perp = uy - uy_par;
    T uz_perp = uz - uz_par;

    // Calculate the acceleration directed parallel to B:
    T ux_next_par = ux_par + qm_dt*Ex_loc_par;
    T uy_next_par = uy_par + qm_dt*Ey_loc_par;
    T uz_next_par = uz_par + qm_dt*Ez_loc_par;

    // Calculate the intermediate gamma:
    T u_square = ux*ux + uy*uy + uz*uz + 1e-16;
    T u_prime_abs = u_square + qm_dt / 2 * (ux*Ex_loc + uy*Ey_loc + uz*Ez_loc);
    T gamma_prime = sqrt(1 + u_prime_abs*u_prime_abs / u_square);

    // Calculate effective perpendicular E and rotaion angle:
    T Ex_prime = gamma_prime * Ex_loc_perp / B_abs;
    T Ey_prime = gamma_prime * Ey_loc_perp / B_abs;
    T Ez_prime = gamma_prime * Ez_loc_perp / B_abs;
    T teta = qm_dt * B_abs / gamma_prime;

    // Calculate the final momentum:
    T cos_val = cos(teta);
    T sin_val = sin(teta);
    ux = (ux_next_par + ux_perp * cos_val
          + (Ey_prime*Bz_unit - Ez_prime*By_unit) * (1 - cos_val)
          + (Ex_prime + uy_perp*Bz_unit - uz_perp*By_unit) * sin_val);
    uy = (uy_next_par + uy_perp * cos_val
          + (Ez_prime*Bx_unit - Ex_prime*Bz_unit) * (1 - cos_val)
          + (Ey_prime + uz_perp*Bx_unit - ux_perp*Bz_unit) * sin_val);
    uz = (uz_next_par + uz_perp * cos_val
          + (Ex_prime*By_unit - Ey_prime*Bx_unit) * (1 - cos_val)
          + (Ez_prime + ux_perp*By_unit - uy_perp*Bx_unit) * sin_val);
"""


# E cross B pusher by Takayuki Umeda and Riku Ozaki in cos / sin form 
# with Gamma = gamma_m from Boris pusher.
# https://doi.org/10.1016/j.jcp.2022.111694
# https://doi.org/10.1186/s40623-023-01902-8
update_momentum_umeda_ozaki = """
    //  Calculate intermediate gamma
    T qm_dt = q_m * dt;
    T qm_dt_2 = qm_dt / 2;
    T ux_m = ux + qm_dt_2*Ex_loc;
    T uy_m = uy + qm_dt_2*Ey_loc;
    T uz_m = uz + qm_dt_2*Ez_loc;
    T inv_gammam_m = 1 / sqrt(1 + ux_m*ux_m + uy_m*uy_m + uz_m*uz_m);

    // Calculate E cros B drift values:
    T B_square = Bx_loc*Bx_loc + By_loc*By_loc + Bz_loc*Bz_loc + 1e-16;
    T B_abs = sqrt(B_square);
    T vx_drift = (Ey_loc*Bz_loc - Ez_loc*By_loc) / B_square;
    T vy_drift = (Ez_loc*Bx_loc - Ex_loc*Bz_loc) / B_square;
    T vz_drift = (Ex_loc*By_loc - Ey_loc*Bx_loc) / B_square;
    T v_drift_square = vx_drift*vx_drift + vy_drift*vy_drift + vz_drift*vz_drift;
    T gamma_drift, gamma_boost, teta, cos_val, sin_val, f1, f2, f3, f4;
    if (v_drift_square < 1 - 1e-15){
        gamma_drift = 1 / sqrt(1 - v_drift_square);
        gamma_boost = 
            gamma_drift * (gamma - vx_drift*ux - vy_drift*uy - vz_drift*uz);

        // Calculate rotation angle 
        teta = qm_dt * B_abs * inv_gammam_m / gamma_drift;
        cos_val = 1 - cos(teta);
        sin_val = sin(teta);

        // Calculate intermideate values
        f1 = sin_val * gamma_drift / B_abs;
        f2 = cos_val / B_square;
        f3 = cos_val * gamma_boost * gamma_drift;
        f4 = qm_dt - sin_val * gamma * gamma_drift / B_abs;
    } else{ 
        if (v_drift_square > 1 + 1e-15){
            gamma_drift = 1 / sqrt(v_drift_square - 1);
            gamma_boost = 
                gamma_drift * (gamma - vx_drift*ux - vy_drift*uy - vz_drift*uz);

            // Calculate rotation angle
            teta = qm_dt * B_abs * inv_gammam_m / gamma_drift;
            cos_val = 1 - cosh(teta);
            sin_val = sinh(teta);

            // Calculate intermideate values
            f1 = sin_val * gamma_drift / B_abs;
            f2 = cos_val / B_square;
            f3 = -cos_val * gamma_boost * gamma_drift;
            f4 = qm_dt - sin_val * gamma * gamma_drift / B_abs;
        } else{
            // Calculate intermideate values
            f1 = qm_dt * inv_gammam_m;
            f2 = f3 = 0;
            f4 = qm_dt * (1 - gamma*inv_gammam_m);
        }
    }

    // Calculate final momentum
    T u_cross_B_x = uy*Bz_loc - uz*By_loc;
    T u_cross_B_y = uz*Bx_loc - ux*Bz_loc;
    T u_cross_B_z = ux*By_loc - uy*Bx_loc;
    ux += (qm_dt * Ex_loc 
           + f1 * u_cross_B_x
           + f2 * (u_cross_B_y*Bz_loc - u_cross_B_z*By_loc)
           + f3 * vx_drift
           + f4 * (vy_drift*Bz_loc - vz_drift*By_loc));
    uy += (qm_dt * Ey_loc 
           + f1 * u_cross_B_y
           + f2 * (u_cross_B_z*Bx_loc - u_cross_B_x*Bz_loc)
           + f3 * vy_drift
           + f4 * (vz_drift*Bx_loc - vx_drift*Bz_loc));
    uz += (qm_dt * Ez_loc 
           + f1 * u_cross_B_z
           + f2 * (u_cross_B_x*By_loc - u_cross_B_y*Bx_loc)
           + f3 * vz_drift
           + f4 * (vx_drift*By_loc - vy_drift*Bx_loc));
"""

# Vay pusher. 
# https://doi.org/10.1063/1.2837054
update_momentum_vay = """
    T qm_dt_2 = q_m * dt / 2.;
    // Calculate intermediate momentum: 
    T vx = ux / gamma;
    T vy = uy / gamma;
    T vz = uz / gamma;
    T ux_prime = ux + qm_dt_2 * (2*Ex_loc + vy*Bz_loc - vz*By_loc);
    T uy_prime = uy + qm_dt_2 * (2*Ey_loc + vz*Bx_loc - vx*Bz_loc);
    T uz_prime = uz + qm_dt_2 * (2*Ez_loc + vx*By_loc - vy*Bx_loc);

    // Calculate intermediate  gamma:
    T bx = qm_dt_2 * Bx_loc;
    T by = qm_dt_2 * By_loc;
    T bz = qm_dt_2 * Bz_loc;
    T b_sq = bx*bx + by*by + bz*bz;

    T gamma_pr_square = 1. 
        + ux_prime*ux_prime + uy_prime*uy_prime + uz_prime*uz_prime;
    T sigma_2 = (gamma_pr_square - b_sq) / 2;
    T b_dot_u = bx*ux_prime + by*uy_prime + bz*uz_prime;
    T gamma_new = sqrt(sigma_2 + sqrt(sigma_2*sigma_2 + b_sq + b_dot_u*b_dot_u));

    // Calculate auxiliary values:
    T tx = bx / gamma_new;
    T ty = by / gamma_new;
    T tz = bz / gamma_new;
    T s = 1. / (1. + tx*tx + ty*ty + tz*tz);
    
    T t_dot_p_prime = tx*ux_prime + ty*uy_prime + tz*uz_prime;

    // Compute a new momentum at full time step:
    ux = s * (ux_prime + tx*t_dot_p_prime + uy_prime*tz - uz_prime*ty);
    uy = s * (uy_prime + ty*t_dot_p_prime + uz_prime*tx - ux_prime*tz);
    uz = s * (uz_prime + tz*t_dot_p_prime + ux_prime*ty - uy_prime*tx);
"""

# Integrator for the Lapenta–Markidis momentum update from paper: 
# http://dx.doi.org/10.1063/1.3602216
# Explicit solution from paper:
# https://doi.org/10.3847/1538-4365/acefba
# with numerical solution of gamma equation.
update_momentum_lapenta_markidis = """
    T qm_dt_2 = q_m * dt / 2;
    T ex = qm_dt_2 * Ex_loc;
    T ey = qm_dt_2 * Ey_loc;
    T ez = qm_dt_2 * Ez_loc;

    T ux_m = ux + ex;
    T uy_m = uy + ey;
    T uz_m = uz + ez;

    T bx = qm_dt_2 * Bx_loc;
    T by = qm_dt_2 * By_loc;
    T bz = qm_dt_2 * Bz_loc;

    T b_sq = bx*bx + by*by + bz*bz;
    T b_dot_p_m = bx*ux_m + by*uy_m + bz*uz_m;
    T p_m_cross_b_x = uy_m*bz - uz_m*by;
    T p_m_cross_b_y = uz_m*bx - ux_m*bz;
    T p_m_cross_b_z = ux_m*by - uy_m*bx;
    T e_dot_b = ex*bx + ey*by + ez*bz;
    T e_dot_p_m = ex*ux_m + ey*uy_m + ez*uz_m;

    T kappa = e_dot_p_m - b_sq;
    T eta = p_m_cross_b_x*ex + p_m_cross_b_y*ey + p_m_cross_b_z*ez + b_sq*gamma;
    T zeta = b_dot_p_m * e_dot_b;
    T gamma_prev = gamma - ((kappa * gamma*gamma + eta*gamma + zeta) 
                            / (2*kappa*gamma + eta - gamma*gamma*gamma));
    T gamma_next = gamma_prev - (f(gamma_prev, -1, gamma, kappa, eta, zeta)
                                 / f_prime(gamma_prev, -1, gamma, kappa, eta));
    while(abs(gamma_next - gamma_prev) / gamma_next > 1e-13){
        gamma_prev = gamma_next;
        gamma_next = gamma_prev - (f(gamma_prev, -1, gamma, kappa, eta, zeta)
                                   / f_prime(gamma_prev, -1, gamma, kappa, eta));
    }
    T gamma_aver = gamma_next;
    T denum = 1 + b_sq/gamma_aver/gamma_aver;
    T ux_aver = (ux_m + b_dot_p_m * bx / gamma_aver / gamma_aver
                 + p_m_cross_b_x / gamma_aver) / denum;
    T uy_aver = (uy_m + b_dot_p_m * by / gamma_aver / gamma_aver
                 + p_m_cross_b_y / gamma_aver) / denum;
    T uz_aver = (uz_m + b_dot_p_m * bz / gamma_aver / gamma_aver
                 + p_m_cross_b_z / gamma_aver) / denum;
    ux = 2 * ux_aver - ux;
    uy = 2 * uy_aver - uy;
    uz = 2 * uz_aver - uz;
"""

# help functions for Lapenta–Markidis
f = """
__device__ inline T f(T x, T a, T b, T c, T d, T e) {
    return a*x*x*x*x + b*x*x*x + c*x*x + d*x + e;
}
"""

f_prime = """
__device__ inline T f_prime(T x, T a, T b, T c, T d) {
    return 4*a*x*x*x + 3*b*x*x + 2*c*x + d;
}
"""

def get_beam_pusher_cupy(integration_method):
    """
    Generate a function to integrate the beam particles for the GPU.

    Parameters
    ----------
    integration_method : str
        Name of the method for integrating the equations of motion.

    Returns
    -------
    push_beam : func
        Function for beam particles integration.
    """
    preamble=weight1_cupy+weight4_cupy
    import cupy as cp
    if integration_method == 'higuera-cary':
        update_momentum = update_momentum_higuera_cary
    elif integration_method == 'runge-kutta':
        update_momentum = update_momentum_runge_kutta
    elif integration_method == 'boris':
        update_momentum = update_momentum_boris
    elif integration_method == 'exact-gyration':
        update_momentum = update_momentum_exact_gyration
    elif integration_method == 'boris-tg':
        update_momentum = update_momentum_boris_tg
    elif integration_method == 'vd1':
        update_momentum = update_momentum_vd1
    elif integration_method == 'umeda-ozaki':
        update_momentum = update_momentum_umeda_ozaki
    elif integration_method == 'vay':
        update_momentum = update_momentum_vay
    elif integration_method == 'lapenta-markidis':
        preamble += f+f_prime
        update_momentum = update_momentum_lapenta_markidis
    else:
        update_momentum = None
        raise ValueError('Unavalibe momentum integrator: {integration_method}.')

    return cp.ElementwiseKernel(
        in_params="""
        float64 xi_step_size, float64 r_max, float64 plasma_slice_idx,
        float64 grid_step_size, float64 grid_steps,
        raw T Ex_prev,  raw T Ey_prev,  raw T Ez_prev,
        raw T Bx_prev,  raw T By_prev,  raw T Bz_prev,
        raw T Ex, raw T Ey, raw T Ez,
        raw T Bx, raw T By, raw T Bz,
        raw T beam_q_m, raw T beam_dt
        """,
        out_params="""
        raw int64 out_remaining_steps, raw int64 out_id,
        raw T out_x, raw T out_y, raw T out_xi,
        raw T out_ux, raw T out_uy, raw T out_uz,
        raw bool lost_idxes, raw bool moved_idxes, raw bool fell_idxes
        """,
        operation="""
        const double plasma_slice_xi = -plasma_slice_idx * xi_step_size;

        const double q_m = beam_q_m[i];
        const double dt = beam_dt[i];
        T x = out_x[i];
        T y = out_y[i];
        T xi = out_xi[i];
        T ux = out_ux[i];
        T uy = out_uy[i];
        T uz = out_uz[i];

        while (out_remaining_steps[i] > 0) {
            // We use "synchronized" version of the leapfrog (DKD version).
            // See details in https://doi.org/10.1086/301102.
            // Calculate the position at the half time step.
            T gamma = sqrt(1. + ux*ux + uy*uy + uz*uz);
            T dt_ef = dt / 2. / gamma;
            T x_half  = x  + dt_ef * ux ;
            T y_half  = y  + dt_ef * uy ;
            T xi_half = xi + dt_ef * uz - dt / 2.;

            if (xi_half < plasma_slice_xi) {
                // Particle will be pushed later.
                fell_idxes[i] = true;
                break;
            }

            if (x_half*x_half + y_half*y_half >= r_max*r_max) {
                // Particle hit the wall and is now lost.
                out_x[i] = x_half, out_y[i] = y_half, out_xi[i] = xi_half;
                out_id[i] *= -1;
                lost_idxes[i] = true;
                out_remaining_steps[i] = 0;
                break;
            }

            // Interpolate fields from the grid on the particle position:
            T x_h = x_half / (T) grid_step_size + 0.5;
            T y_h = y_half / (T) grid_step_size + 0.5;
            T x_loc = x_h - floor(x_h) - 0.5;
            T y_loc = y_h - floor(y_h) - 0.5;
            T xi_loc = ((T) xi_half - plasma_slice_xi) / (T) xi_step_size;
            int ix = floor(x_h) + floor(grid_steps / 2);
            int iy = floor(y_h) + floor(grid_steps / 2);

            T Ex_loc = 0, Ey_loc = 0, Ez_loc = 0;
            T Bx_loc = 0, By_loc = 0, Bz_loc = 0;
            for (int kx = -2; kx <= 2; kx++) {
                const T wx = weight4(x_loc, kx);
                for (int ky = -2; ky <= 2; ky++) {
                    const T w  = wx * weight4(y_loc,  ky);
                    const T w0 = w  * weight1(xi_loc, 0);
                    const T w1 = w  * weight1(xi_loc, 1);
                    const int idx = (iy + ky) + (int) grid_steps * (ix + kx);

                    Ex_loc += Ex_prev[idx] * w0 + Ex[idx] * w1;
                    Bx_loc += Bx_prev[idx] * w0 + Bx[idx] * w1;
                    Ey_loc += Ey_prev[idx] * w0 + Ey[idx] * w1;
                    By_loc += By_prev[idx] * w0 + By[idx] * w1;
                    Ez_loc += Ez_prev[idx] * w0 + Ez[idx] * w1;
                    Bz_loc += Bz_prev[idx] * w0 + Bz[idx] * w1;
                }
            }
            """
            + update_momentum
            + """
            // Calculate a new position at full time step:
            gamma = sqrt(1. + ux*ux + uy*uy + uz*uz);
            dt_ef = dt / 2. / gamma;
            x  = x_half  + dt_ef * ux;
            y  = y_half  + dt_ef * uy;
            xi = xi_half + dt_ef * uz - dt / 2.;
            if (x*x + y*y >= r_max*r_max) {
                // Particle hit the wall and is now lost
                out_id[i] *= -1;
                lost_idxes[i] = true;
                out_remaining_steps[i] = 0;
                break;
            }

            out_remaining_steps[i] -= 1;
        }
        out_x[i]  = x;
        out_y[i]  = y;
        out_xi[i] = xi;
        out_ux[i] = ux;
        out_uy[i] = uy;
        out_uz[i] = uz;

        if (lost_idxes[i] == false){
            if (out_remaining_steps[i] == 0 && out_xi[i] > plasma_slice_xi ){
                moved_idxes[i] = true;
            }else{
                fell_idxes[i] = true;
            }
        }
        """,
        name='push_beam_cupy', preamble=preamble,
        no_return=True
    )