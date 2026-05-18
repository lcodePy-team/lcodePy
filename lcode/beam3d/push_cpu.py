import numba as nb
import numpy as np

from math import sqrt, pow, floor

from ..beam3d.weights import weight1, weight4


@nb.njit
def is_lost(x, y, r_max):
    return x ** 2 + y ** 2 >= r_max ** 2


@nb.njit(forceinline=True)
def update_momentum_runge_kutta(dt, 
                                Ex_loc, Ey_loc, Ez_loc, 
                                Bx_loc, By_loc, Bz_loc, 
                                q_m, gamma, ux, uy, uz):
    '''
    4-order Runge-Kutta method. 
    '''
    # First half electric field acceleration
    qm_dt = q_m * dt
    Ex_loc *= qm_dt
    Ey_loc *= qm_dt
    Ez_loc *= qm_dt
    Bx_loc *= qm_dt
    By_loc *= qm_dt
    Bz_loc *= qm_dt

    vx = ux / gamma
    vy = uy / gamma
    vz = uz / gamma
    k1_x = Ex_loc + vy*Bz_loc - vz*By_loc
    k1_y = Ey_loc + vz*Bx_loc - vx*Bz_loc
    k1_z = Ez_loc + vx*By_loc - vy*Bx_loc

    ux_half = ux + k1_x/2
    uy_half = uy + k1_y/2
    uz_half = uz + k1_z/2
    gamma_half = sqrt(1 + ux_half**2 + uy_half**2 + uz_half**2)
    vx_half = ux_half / gamma_half
    vy_half = uy_half / gamma_half
    vz_half = uz_half / gamma_half
    k2_x = Ex_loc + vy_half*Bz_loc - vz_half*By_loc
    k2_y = Ey_loc + vz_half*Bx_loc - vx_half*Bz_loc
    k2_z = Ez_loc + vx_half*By_loc - vy_half*Bx_loc

    ux_half = ux + k2_x/2
    uy_half = uy + k2_y/2
    uz_half = uz + k2_z/2
    gamma_half = sqrt(1 + ux_half**2 + uy_half**2 + uz_half**2)
    vx_half = ux_half / gamma_half
    vy_half = uy_half / gamma_half
    vz_half = uz_half / gamma_half
    k3_x = Ex_loc + vy_half*Bz_loc - vz_half*By_loc
    k3_y = Ey_loc + vz_half*Bx_loc - vx_half*Bz_loc
    k3_z = Ez_loc + vx_half*By_loc - vy_half*Bx_loc
    
    ux_half = ux + k3_x
    uy_half = uy + k3_y
    uz_half = uz + k3_z
    gamma_half = sqrt(1 + ux_half**2 + uy_half**2 + uz_half**2)
    vx_half = ux_half / gamma_half
    vy_half = uy_half / gamma_half
    vz_half = uz_half / gamma_half
    k4_x = Ex_loc + vy_half*Bz_loc - vz_half*By_loc
    k4_y = Ey_loc + vz_half*Bx_loc - vx_half*Bz_loc
    k4_z = Ez_loc + vx_half*By_loc - vy_half*Bx_loc

    ux += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
    uy += (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
    uz += (k1_z + 2*k2_z + 2*k3_z + k4_z) / 6

    return ux, uy, uz

@nb.njit(forceinline=True)
def update_momentum_boris(dt, 
                          Ex_loc, Ey_loc, Ez_loc, 
                          Bx_loc, By_loc, Bz_loc, 
                          q_m, gamma, ux, uy, uz):
    '''
    Boris pusher: J.P. Boris, Relativistic plasma simulation-optimization 
    of a hybrid code, in: Proceedings of 4th Conference on Numerical Simulation 
    of Plasmas, Naval  Research Laboratory, Washington D.C., 1970, pp. 3–67
    '''
    # First half electric field acceleration
    qm_dt_2 = q_m * dt / 2
    ux_m = ux + qm_dt_2*Ex_loc
    uy_m = uy + qm_dt_2*Ey_loc
    uz_m = uz + qm_dt_2*Ez_loc

    # Calculate the gamma for rotation by a magnetic field:
    gamma_m = sqrt(1. + ux_m**2 + uy_m**2 + uz_m**2)

    # Calculate auxiliary values:
    _t = qm_dt_2 / gamma_m 
    tx = _t * Bx_loc
    ty = _t * By_loc
    tz = _t * Bz_loc
    t_sq_pl = 2 / (1. + tx**2 + ty**2 + tz**2)
    sx = tx * t_sq_pl
    sy = ty * t_sq_pl
    sz = tz * t_sq_pl

    # Rotation step + second half electric field acceleration: 
    ux_prime = ux_m + uy_m*tz - uz_m*ty
    uy_prime = uy_m + uz_m*tx - ux_m*tz
    uz_prime = uz_m + ux_m*ty - uy_m*tx
    ux = ux_m + uy_prime*sz - uz_prime*sy + qm_dt_2*Ex_loc
    uy = uy_m + uz_prime*sx - ux_prime*sz + qm_dt_2*Ey_loc
    uz = uz_m + ux_prime*sy - uy_prime*sx + qm_dt_2*Ez_loc

    return ux, uy, uz


@nb.njit(forceinline=True)
def update_momentum_boris_tg(dt, 
                             Ex_loc, Ey_loc, Ez_loc, 
                             Bx_loc, By_loc, Bz_loc,
                             q_m, gamma, ux, uy, uz):
    '''
    Boris pusher: J.P. Boris, Relativistic plasma simulation-optimization 
    of a hybrid code, in: Proceedings of 4th Conference on Numerical Simulation 
    of Plasmas, Naval  Research Laboratory, Washington D.C., 1970, pp. 3–67
    '''
    # First half electric field acceleration
    qm_dt_2 = q_m * dt / 2
    ux_m = ux + qm_dt_2*Ex_loc
    uy_m = uy + qm_dt_2*Ey_loc
    uz_m = uz + qm_dt_2*Ez_loc

    # Calculate the gamma for rotation by a magnetic field:
    gamma_m = sqrt(1. + ux_m**2 + uy_m**2 + uz_m**2)

    # Calculate auxiliary values:
    B_abs = sqrt(Bx_loc**2 + By_loc**2 + Bz_loc**2) + 1e-16
    _t = np.tan(qm_dt_2 * B_abs / gamma_m) / B_abs
    tx = _t * Bx_loc
    ty = _t * By_loc
    tz = _t * Bz_loc
    t_sq_pl = 2 / (1. + tx**2 + ty**2 + tz**2)
    sx = tx * t_sq_pl
    sy = ty * t_sq_pl
    sz = tz * t_sq_pl

    # Rotation step + second half electric field acceleration: 
    ux_prime = ux_m + uy_m*tz - uz_m*ty
    uy_prime = uy_m + uz_m*tx - ux_m*tz
    uz_prime = uz_m + ux_m*ty - uy_m*tx
    ux = ux_m + uy_prime*sz - uz_prime*sy + qm_dt_2*Ex_loc
    uy = uy_m + uz_prime*sx - ux_prime*sz + qm_dt_2*Ey_loc
    uz = uz_m + ux_prime*sy - uy_prime*sx + qm_dt_2*Ez_loc

    return ux, uy, uz

@nb.njit(forceinline=True)
def update_momentum_exact_gyration(dt, 
                                   Ex_loc, Ey_loc, Ez_loc, 
                                   Bx_loc, By_loc, Bz_loc,
                                    q_m, gamma, ux, uy, uz):
    '''
    Exact geration pusher by Seiji Zenitani and Takayuki Umeda.
    https://doi.org/10.1063/1.5051077
    '''
    # First half electric field acceleration:
    qm_dt_2 = q_m * dt / 2
    ux_m = ux + qm_dt_2*Ex_loc
    uy_m = uy + qm_dt_2*Ey_loc
    uz_m = uz + qm_dt_2*Ez_loc

    # Calculate the gamma for rotation by a magnetic field:
    gamma_m = sqrt(1. + ux_m**2 + uy_m**2 + uz_m**2)

    # Calculate auxiliary values:
    B_abs = sqrt(Bx_loc**2 + By_loc**2 + Bz_loc**2) + 1e-16
    Bx_unit = Bx_loc / B_abs
    By_unit = By_loc / B_abs
    Bz_unit = Bz_loc / B_abs
    B_unit_dot_p_m = Bx_unit*ux_m + By_unit*uy_m + Bz_unit*uz_m
    ux_m_par = B_unit_dot_p_m * Bx_unit
    uy_m_par = B_unit_dot_p_m * By_unit
    uz_m_par = B_unit_dot_p_m * Bz_unit
    teta = q_m * dt * B_abs / gamma_m
    sin_val = np.sin(teta)
    cos_val = np.cos(teta)

    # Rotation step + second half electric field acceleration: 
    ux = (ux_m_par + (ux_m - ux_m_par) * cos_val
          + (uy_m*Bz_unit - uz_m*By_unit) * sin_val
          + qm_dt_2 * Ex_loc)
    uy = (uy_m_par + (uy_m - uy_m_par) * cos_val
          + (uz_m*Bx_unit - ux_m*Bz_unit) * sin_val
          + qm_dt_2 * Ey_loc) 
    uz = (uz_m_par + (uz_m - uz_m_par) * cos_val
          + (ux_m*By_unit - uy_m*Bx_unit) * sin_val
          + qm_dt_2 * Ez_loc)

    return ux, uy, uz

@nb.njit(forceinline=True)
def update_momentum_vd1(dt, 
                        Ex_loc, Ey_loc, Ez_loc, 
                        Bx_loc, By_loc, Bz_loc,
                        q_m, gamma, ux, uy, uz):
    '''
    VD1 pusher by K.V. Vshivkov, E.S. Voropaeva, A.A. Efimova
    https://doi.org/10.25743/ICT.2023.282.004
    '''
    qm_dt = q_m * dt
    # Calculate parallel abd perpendicular values:
    B_abs = sqrt(Bx_loc**2 + By_loc**2 + Bz_loc**2) + 1e-16
    Bx_unit = Bx_loc / B_abs
    By_unit = By_loc / B_abs
    Bz_unit = Bz_loc / B_abs
    B_unit_dot_u = Bx_unit*ux + By_unit*uy + Bz_unit*uz
    B_unit_dot_E = Bx_unit*Ex_loc + By_unit*Ey_loc + Bz_unit*Ez_loc
    Ex_loc_par = B_unit_dot_E * Bx_unit
    Ey_loc_par = B_unit_dot_E * By_unit
    Ez_loc_par = B_unit_dot_E * Bz_unit
    Ex_loc_perp = Ex_loc - Ex_loc_par
    Ey_loc_perp = Ey_loc - Ey_loc_par
    Ez_loc_perp = Ez_loc - Ez_loc_par
    ux_par = B_unit_dot_u * Bx_unit
    uy_par = B_unit_dot_u * By_unit
    uz_par = B_unit_dot_u * Bz_unit
    ux_perp = ux - ux_par
    uy_perp = uy - uy_par
    uz_perp = uz - uz_par

    # Calculate the acceleration directed parallel to B:
    ux_next_par = ux_par + qm_dt*Ex_loc_par
    uy_next_par = uy_par + qm_dt*Ey_loc_par
    uz_next_par = uz_par + qm_dt*Ez_loc_par

    # Calculate the intermediate gamma:
    u_square = ux**2 + uy**2 + uz**2 + 1e-16
    u_prime_abs = u_square + qm_dt / 2 * (ux*Ex_loc + uy*Ey_loc + uz*Ez_loc)
    gamma_prime = sqrt(1 + u_prime_abs**2 / u_square)

    # Calculate effective perpendicular E and rotaion angle:
    Ex_prime = gamma_prime * Ex_loc_perp / B_abs
    Ey_prime = gamma_prime * Ey_loc_perp / B_abs
    Ez_prime = gamma_prime * Ez_loc_perp / B_abs
    teta = qm_dt * B_abs / gamma_prime

    # Calculate the final momentum:
    cos_val = np.cos(teta)
    sin_val = np.sin(teta)
    ux = (ux_next_par + ux_perp * cos_val
          + (Ey_prime*Bz_unit - Ez_prime*By_unit) * (1 - cos_val)
          + (Ex_prime + uy_perp*Bz_unit - uz_perp*By_unit) * sin_val)
    uy = (uy_next_par + uy_perp * cos_val
          + (Ez_prime*Bx_unit - Ex_prime*Bz_unit) * (1 - cos_val)
          + (Ey_prime + uz_perp*Bx_unit - ux_perp*Bz_unit) * sin_val)
    uz = (uz_next_par + uz_perp * cos_val
          + (Ex_prime*By_unit - Ey_prime*Bx_unit) * (1 - cos_val)
          + (Ez_prime + ux_perp*By_unit - uy_perp*Bx_unit) * sin_val)

    return ux, uy, uz

@nb.njit(forceinline=True)
def update_momentum_umeda_ozaki(dt, 
                                Ex_loc, Ey_loc, Ez_loc, 
                                Bx_loc, By_loc, Bz_loc,
                                q_m, gamma, ux, uy, uz):
    '''
    E cross B pusher by Takayuki Umeda and Riku Ozaki in cos / sin form 
    with Gamma = gamma_m from Boris pusher.
    https://doi.org/10.1016/j.jcp.2022.111694
    https://doi.org/10.1186/s40623-023-01902-8
    '''
    #  Calculate intermediate gamma
    qm_dt = q_m * dt
    qm_dt_2 = qm_dt / 2
    ux_m = ux + qm_dt_2*Ex_loc
    uy_m = uy + qm_dt_2*Ey_loc
    uz_m = uz + qm_dt_2*Ez_loc
    inv_gammam_m = 1 / sqrt(1 + ux_m**2 + uy_m**2 + uz_m**2)

    # Calculate E cros B drift values:
    B_square = Bx_loc**2 + By_loc**2 + Bz_loc**2 + 1e-16
    B_abs = sqrt(B_square)
    vx_drift = (Ey_loc*Bz_loc - Ez_loc*By_loc) / B_square
    vy_drift = (Ez_loc*Bx_loc - Ex_loc*Bz_loc) / B_square
    vz_drift = (Ex_loc*By_loc - Ey_loc*Bx_loc) / B_square
    v_drift_square = vx_drift**2 + vy_drift**2 + vz_drift**2
    if v_drift_square < 1 - 1e-15:
        gamma_drift = 1 / sqrt(1 - v_drift_square)
        gamma_boost = gamma_drift * (gamma
                                     - vx_drift*ux - vy_drift*uy - vz_drift*uz)

        # Calculate rotation angle 
        teta = qm_dt * B_abs * inv_gammam_m / gamma_drift
        cos_val = 1 - np.cos(teta)
        sin_val = np.sin(teta)

        # Calculate intermideate values
        f1 = sin_val * gamma_drift / B_abs
        f2 = cos_val / B_square
        f3 = cos_val * gamma_boost * gamma_drift
        f4 = qm_dt - sin_val * gamma * gamma_drift / B_abs
    elif v_drift_square > 1 + 1e-15:
        gamma_drift = 1 / sqrt(v_drift_square - 1)
        gamma_boost = gamma_drift * (gamma
                                     - vx_drift*ux - vy_drift*uy - vz_drift*uz)

        # Calculate rotation angle
        teta = qm_dt * B_abs * inv_gammam_m / gamma_drift
        cos_val = 1 - np.cosh(teta)
        sin_val = np.sinh(teta)

        # Calculate intermideate values
        f1 = sin_val * gamma_drift / B_abs
        f2 = cos_val / B_square
        f3 = -cos_val * gamma_boost * gamma_drift
        f4 = qm_dt - sin_val * gamma * gamma_drift / B_abs
    else:
        # Calculate intermideate values
        f1 = qm_dt * inv_gammam_m
        f2 = f3 = 0
        f4 = qm_dt * (1 - gamma*inv_gammam_m)

    # Calculate final momentum
    u_cross_B_x = uy*Bz_loc - uz*By_loc 
    u_cross_B_y = uz*Bx_loc - ux*Bz_loc 
    u_cross_B_z = ux*By_loc - uy*Bx_loc 
    ux += (qm_dt * Ex_loc 
           + f1 * u_cross_B_x
           + f2 * (u_cross_B_y*Bz_loc - u_cross_B_z*By_loc)
           + f3 * vx_drift
           + f4 * (vy_drift*Bz_loc - vz_drift*By_loc))
    uy += (qm_dt * Ey_loc 
           + f1 * u_cross_B_y
           + f2 * (u_cross_B_z*Bx_loc - u_cross_B_x*Bz_loc)
           + f3 * vy_drift
           + f4 * (vz_drift*Bx_loc - vx_drift*Bz_loc))
    uz += (qm_dt * Ez_loc 
           + f1 * u_cross_B_z
           + f2 * (u_cross_B_x*By_loc - u_cross_B_y*Bx_loc)
           + f3 * vz_drift
           + f4 * (vx_drift*By_loc - vy_drift*Bx_loc))

    return ux, uy, uz

@nb.njit(forceinline=True)
def update_momentum_vay(dt, 
                        Ex_loc, Ey_loc, Ez_loc, 
                        Bx_loc, By_loc, Bz_loc,
                        q_m, gamma, ux, uy, uz):
    '''
    Vay pusher. 
    https://doi.org/10.1063/1.2837054
    '''
    qm_dt_2 = q_m * dt / 2.
    # Calculate intermediate momentum: 
    vx = ux / gamma
    vy = uy / gamma
    vz = uz / gamma
    ux_prime = ux + qm_dt_2 * (2*Ex_loc + vy*Bz_loc - vz*By_loc)
    uy_prime = uy + qm_dt_2 * (2*Ey_loc + vz*Bx_loc - vx*Bz_loc)
    uz_prime = uz + qm_dt_2 * (2*Ez_loc + vx*By_loc - vy*Bx_loc)

    # Calculate intermediate  gamma:
    bx = qm_dt_2 * Bx_loc
    by = qm_dt_2 * By_loc
    bz = qm_dt_2 * Bz_loc
    b_sq = bx**2 + by**2 + bz**2

    gamma_pr_square = 1. + ux_prime**2 + uy_prime**2 + uz_prime**2
    sigma_2 = (gamma_pr_square - b_sq) / 2
    gamma_new = sqrt(sigma_2 
                     + sqrt(sigma_2**2 + b_sq
                            + (bx*ux_prime + by*uy_prime + bz*uz_prime)**2))

    # Calculate auxiliary values:
    tx = bx / gamma_new
    ty = by / gamma_new
    tz = bz / gamma_new
    s = 1. / (1. + tx**2 + ty**2 + tz**2)
    
    t_dot_p_prime = tx*ux_prime + ty*uy_prime + tz*uz_prime

    # Compute a new momentum at full time step:
    ux = s * (ux_prime + tx*t_dot_p_prime + uy_prime*tz - uz_prime*ty)
    uy = s * (uy_prime + ty*t_dot_p_prime + uz_prime*tx - ux_prime*tz)
    uz = s * (uz_prime + tz*t_dot_p_prime + ux_prime*ty - uy_prime*tx)

    return ux, uy, uz

@nb.njit(forceinline=True)
def update_momentum_higuera_cary(dt, 
                                 Ex_loc, Ey_loc, Ez_loc, 
                                 Bx_loc, By_loc, Bz_loc,
                                 q_m, gamma, ux, uy, uz):
    '''
    Higuera-Cary pusher. 
    https://doi.org/10.1063/1.4979989
    '''
    qm_dt_2 = q_m * dt / 2.
    # First half electric field acceleration:
    ux_m = ux + qm_dt_2 * Ex_loc
    uy_m = uy + qm_dt_2 * Ey_loc
    uz_m = uz + qm_dt_2 * Ez_loc

    # Calculate the gamma for rotation by a magnetic field:
    gamma_m_sq = 1. + ux_m**2 + uy_m**2 + uz_m**2

    bx = qm_dt_2 * Bx_loc
    by = qm_dt_2 * By_loc
    bz = qm_dt_2 * Bz_loc
    b_sq = bx**2 + by**2 + bz**2
    sigma_2 = (gamma_m_sq - b_sq) / 2
    gamma_new = sqrt(sigma_2
                     + sqrt((sigma_2)**2 + b_sq
                            + (bx*ux_m + by*uy_m + bz*uz_m)**2))

    # Calculate auxiliary values:
    tx = bx / gamma_new
    ty = by / gamma_new
    tz = bz / gamma_new
    t_sq_pl = 2 / (1. + tx**2 + ty**2 + tz**2)
    sx = tx * t_sq_pl
    sy = ty * t_sq_pl
    sz = tz * t_sq_pl

    # Rotation step + second half electric field acceleration: 
    ux_prime = ux_m + uy_m*tz - uz_m*ty
    uy_prime = uy_m + uz_m*tx - ux_m*tz
    uz_prime = uz_m + ux_m*ty - uy_m*tx
    ux = ux_m + uy_prime*sz - uz_prime*sy + qm_dt_2*Ex_loc
    uy = uy_m + uz_prime*sx - ux_prime*sz + qm_dt_2*Ey_loc
    uz = uz_m + ux_prime*sy - uy_prime*sx + qm_dt_2*Ez_loc

    return ux, uy, uz


@nb.njit(forceinline=True)
def f(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x +e

@nb.njit(forceinline=True)
def f_prime(x, a, b, c, d):
    return 4*a*x**3 + 3*b*x**2 + 2*c*x + d

@nb.njit(forceinline=True)
def update_momentum_lapenta_markidis(dt, 
                                     Ex_loc, Ey_loc, Ez_loc, 
                                     Bx_loc, By_loc, Bz_loc,
                                     q_m, gamma, ux, uy, uz):
    '''
    Integrator for the Lapenta–Markidis momentum update from paper: 
    http://dx.doi.org/10.1063/1.3602216
    Explicit solution from paper:
    https://doi.org/10.3847/1538-4365/acefba
    with numerical solution of gamma equation.
    '''
    qm_dt_2 = q_m * dt / 2
    ex = qm_dt_2 * Ex_loc
    ey = qm_dt_2 * Ey_loc
    ez = qm_dt_2 * Ez_loc

    ux_m = ux + ex
    uy_m = uy + ey
    uz_m = uz + ez

    bx = qm_dt_2 * Bx_loc
    by = qm_dt_2 * By_loc
    bz = qm_dt_2 * Bz_loc

    b_sq = bx**2 + by**2 + bz**2
    b_dot_p_m = bx*ux_m + by*uy_m + bz*uz_m
    p_m_cross_b_x = uy_m*bz - uz_m*by
    p_m_cross_b_y = uz_m*bx - ux_m*bz
    p_m_cross_b_z = ux_m*by - uy_m*bx
    e_dot_b = ex*bx + ey*by + ez*bz
    e_dot_p_m = ex*ux_m + ey*uy_m + ez*uz_m

    kappa = e_dot_p_m - b_sq
    eta = p_m_cross_b_x*ex + p_m_cross_b_y*ey + p_m_cross_b_z*ez + b_sq*gamma
    zeta = b_dot_p_m * e_dot_b
    gamma_prev = gamma - ((kappa * gamma**2 + eta*gamma + zeta) 
                          / (2*kappa*gamma + eta - gamma**3))
    gamma_next = gamma_prev - (f(gamma_prev, -1, gamma, kappa, eta, zeta)
                               / f_prime(gamma_prev, -1, gamma, kappa, eta))
    while(abs(gamma_next - gamma_prev) / gamma_next > 1e-13):
        gamma_prev = gamma_next
        gamma_next = gamma_prev - (f(gamma_prev, -1, gamma, kappa, eta, zeta)
                                   / f_prime(gamma_prev, -1, gamma, kappa, eta))
    gamma_aver = gamma_next
    denum = 1 + b_sq/gamma_aver**2
    ux_aver = (ux_m + b_dot_p_m * bx / gamma_aver**2
               + p_m_cross_b_x / gamma_aver) / denum
    uy_aver = (uy_m + b_dot_p_m * by / gamma_aver**2
               + p_m_cross_b_y / gamma_aver) / denum
    uz_aver = (uz_m + b_dot_p_m * bz / gamma_aver**2
               + p_m_cross_b_z / gamma_aver) / denum
    ux = 2 * ux_aver - ux
    uy = 2 * uy_aver - uy
    uz = 2 * uz_aver - uz

    return ux, uy, uz






def get_beam_pusher_numba(integration_method):
    """
    Generate a function to integrate the beam particles for the CPU.

    Parameters
    ----------
    integration_method : str
        Name of the scheme for integrating the equations of motion.

    Returns
    -------
    push_beam : func
        Function for beam particles integration.
    """
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
        update_momentum = update_momentum_lapenta_markidis
    else:
        update_momentum = None
        raise ValueError('Unavalibe momentum integrator: {integration_method}.')


    @nb.njit(error_model='numpy')
    def push_beam_numba(
        xi_step_size, lost_radius, plasma_slice_idx, grid_step_size, grid_steps,
        Ex_prev,  Ey_prev,  Ez_prev,  Bx_prev,  By_prev,  Bz_prev,
        Ex, Ey, Ez, Bx, By, Bz,
        beam_q_m, beam_dt,
        remaining_steps, id, beam_x, beam_y, beam_xi, beam_ux, beam_uy, beam_uz,
        lost_idxes, moved_idxes, fell_idxes, size):
        """
        Push particles as far as possible within
        plasma_slice_idx and plasma_slice_idx - 1.
        """
        plasma_slice_xi = -plasma_slice_idx * xi_step_size
        for i in nb.prange(beam_x.shape[0]):
            q_m, dt = beam_q_m[i], beam_dt[i]
            x, y, xi = beam_x[i], beam_y[i], beam_xi[i]
            ux, uy, uz = beam_ux[i], beam_uy[i], beam_uz[i]
            while remaining_steps[i] > 0:
                # We use “synchronized” version of the leapfrog (DKD version).
                # See details in https://doi.org/10.1086/301102.
                # Calculate the position at the half time step.
                gamma = np.sqrt(1. + ux**2 + uy**2 + uz**2)
                dt_ef = dt / 2 / gamma
                x_half = x + dt_ef * ux
                y_half = y + dt_ef * uy
                xi_half = xi + dt_ef * uz - dt / 2
                if xi_half < plasma_slice_xi:
                    fell_idxes[i] = True # Particle will be pushed later.
                    break

                if is_lost(x_half, y_half, lost_radius):
                    beam_x[i], beam_y[i], beam_xi[i] = x_half, y_half, xi_half
                    id[i] *= -1  # Particle hit the wall and is now lost.
                    lost_idxes[i] = True
                    remaining_steps[i] = 0
                    break

                # Interpolate fields.
                x_h = x_half / grid_step_size + 0.5
                y_h = y_half / grid_step_size + 0.5
                x_loc = x_h - floor(x_h) - 0.5
                y_loc = y_h - floor(y_h) - 0.5
                xi_loc = (xi_half - plasma_slice_xi) / xi_step_size
                ix = int(floor(x_h) + grid_steps // 2)
                iy = int(floor(y_h) + grid_steps // 2)
                Ex_loc, Ey_loc, Ez_loc = 0, 0, 0
                Bx_loc, By_loc, Bz_loc = 0, 0, 0
                for kx in range(-2, 3):
                    wx = weight4(x_loc, kx)
                    for ky in range(-2, 3):
                        w = wx * weight4(y_loc, ky)
                        w0 = w * weight1(xi_loc, 0)
                        w1 = w * weight1(xi_loc, 1)
                        idx_x, idx_y = ix + kx, iy + ky
                        # Collect value from a cell and 8 surrounding cells.
                        Ex_loc += Ex_prev[idx_x, idx_y] * w0 + Ex[idx_x, idx_y] * w1
                        Ey_loc += Ey_prev[idx_x, idx_y] * w0 + Ey[idx_x, idx_y] * w1
                        Ez_loc += Ez_prev[idx_x, idx_y] * w0 + Ez[idx_x, idx_y] * w1
                        Bx_loc += Bx_prev[idx_x, idx_y] * w0 + Bx[idx_x, idx_y] * w1
                        By_loc += By_prev[idx_x, idx_y] * w0 + By[idx_x, idx_y] * w1
                        Bz_loc += Bz_prev[idx_x, idx_y] * w0 + Bz[idx_x, idx_y] * w1


                ux, uy, uz = update_momentum(dt,
                                             Ex_loc, Ey_loc, Ez_loc,
                                             Bx_loc, By_loc, Bz_loc,
                                             q_m, gamma, ux, uy, uz)

                # Calculate a new position at full time step:
                gamma = sqrt(1. + ux**2 + uy**2 + uz**2)
                dt_ef = dt / 2 / gamma
                x = x_half + dt_ef * ux
                y = y_half + dt_ef * uy
                xi = xi_half + dt_ef * uz - dt / 2
                
                if is_lost(x, y, lost_radius):
                   id[i] *= -1  # Particle hit the wall and is now lost.
                   lost_idxes[i] = True
                   remaining_steps[i] = 0
                   break

                remaining_steps[i] -= 1

            if not lost_idxes[i]:
                if remaining_steps[i] == 0 and xi > plasma_slice_xi:
                    moved_idxes[i] = True
                else:
                    fell_idxes[i] = True
            
            beam_x[i], beam_y[i], beam_xi[i] = x, y, xi
            beam_ux[i], beam_uy[i], beam_uz[i] = ux, uy, uz

    return push_beam_numba