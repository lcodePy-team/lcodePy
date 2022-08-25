"""Functions to find fields on the next step of plasma evolution."""
import cupy as cp

from ..config.config import Config
from .data import GPUArrays, fields_average


# Solving Laplace equation with Dirichlet boundary conditions (Ez and Phi) #

def dst2d(a):
    """
    Calculate DST-Type1-2D, jury-rigged from anti-symmetrically-padded rFFT.
    """
    assert a.shape[0] == a.shape[1]
    N = a.shape[0]
    #                                    / 0  0  0  0  0  0 \
    #  0  0  0  0                       |  0 /1  2\ 0 -2 -1  |
    #  0 /1  2\ 0   anti-symmetrically  |  0 \3  4/ 0 -4 -3  |
    #  0 \3  4/ 0       padded to       |  0  0  0  0  0  0  |
    #  0  0  0  0                       |  0 -3 -4  0 +4 +3  |
    #                                    \ 0 -1 -2  0 +2 +1 /
    p = cp.zeros((2 * N + 2, 2 * N + 2))
    p[1:N+1, 1:N+1], p[1:N+1, N+2:] = a,             -cp.fliplr(a)
    p[N+2:,  1:N+1], p[N+2:,  N+2:] = -cp.flipud(a), +cp.fliplr(cp.flipud(a))

    # after padding: rFFT-2D, cut out the top-left segment, take -real part
    return -cp.fft.rfft2(p)[1:N+1, 1:N+1].real


def calculate_Ez(grid_step_size, const, currents):
    """
    Calculate Ez as iDST2D(dirichlet_matrix * DST2D(djx/dx + djy/dy)).
    """
    # 0. Calculate RHS (NOTE: it is smaller by 1 on each side).
    # NOTE: use gradient instead if available (cupy doesn't have gradient yet).
    jx, jy = currents.jx, currents.jy

    djx_dx = jx[2:, 1:-1] - jx[:-2, 1:-1]
    djy_dy = jy[1:-1, 2:] - jy[1:-1, :-2]
    rhs_inner = -(djx_dx + djy_dy) / (grid_step_size * 2)  # -?

    # 1. Apply DST-Type1-2D (Discrete Sine Transform Type 1 2D) to the RHS.
    f = dst2d(rhs_inner)

    # 2. Multiply f by the special matrix that does the job and normalizes.
    f *= const.dirichlet_matrix

    # 3. Apply iDST-Type1-2D (Inverse Discrete Sine Transform Type 1 2D).
    #    We don't have to define a separate iDST function, because
    #    unnormalized DST-Type1 is its own inverse, up to a factor 2(N+1)
    #    and we take all scaling matters into account with a single factor
    #    hidden inside dirichlet_matrix.
    Ez_inner = dst2d(f)
    Ez = cp.pad(Ez_inner, 1, 'constant', constant_values=0)

    return Ez


def calculate_Phi(const, currents):
    """
    Calculates Phi as iDST2D(dirichlet_matrix * DST2D(-ro + jz)).
    """
    ro, jz = currents.ro, currents.jz

    rhs_inner = (ro - jz)[1:-1, 1:-1]

    f = dst2d(rhs_inner)

    f *= const.dirichlet_matrix

    Phi_inner = dst2d(f)
    Phi = cp.pad(Phi_inner, 1, 'constant', constant_values=0)

    return Phi


# Solving Laplace or Helmholtz equation with mixed boundary conditions #

def mix2d(a):
    """
    Calculate a DST-DCT-hybrid transform
    (DST in first direction, DCT in second one).
    """
    # NOTE: LCODE 3D uses x as the first direction, thus the confision below.
    M, N = a.shape
    #                                  /(0  1  2  0)-2 -1 \      +---->  x
    #  / 1  2 \                       | (0  3  4  0)-4 -3  |     |      (M)
    #  | 3  4 |  mixed-symmetrically  | (0  5  6  0)-6 -5  |     |
    #  | 5  6 |       padded to       | (0  7  8  0)-8 -7  |     v
    #  \ 7  8 /                       |  0 +5 +6  0 -6 -5  |
    #                                  \ 0 +3 +4  0 -4 -3 /      y (N)
    p = cp.zeros((2 * M + 2, 2 * N - 2))  # wider than before
    p[1:M+1, :N] = a
    p[M+2:2*M+2, :N] = -cp.flipud(a)  # flip to right on drawing above
    p[1:M+1, N-1:2*N-2] = cp.fliplr(a)[:, :-1]  # flip down on drawing above
    p[M+2:2*M+2, N-1:2*N-2] = -cp.flipud(cp.fliplr(a))[:, :-1]
    # Note: the returned array is wider than the input array, it is padded
    # with zeroes (depicted above as a square region marked with round braces).
    return -cp.fft.rfft2(p)[:M+2, :N].imag  # FFT, cut a corner with 0s, -imag


def dx_dy(arr, grid_step_size):
    """
    Calculate x and y derivatives simultaneously (like np.gradient does).
    NOTE: use gradient instead if available (cupy doesn't have gradient yet).
    NOTE: arrays are assumed to have zeros on the perimeter.
    """
    dx, dy = cp.zeros_like(arr), cp.zeros_like(arr)
    dx[1:-1, 1:-1] = arr[2:, 1:-1] - arr[:-2, 1:-1]  # arrays have 0s
    dy[1:-1, 1:-1] = arr[1:-1, 2:] - arr[1:-1, :-2]  # on the perimeter
    return dx / (grid_step_size * 2), dy / (grid_step_size * 2)


def calculate_Ex_Ey_Bx_By(
    grid_step_size, xi_step_size, const, fields, ro_beam_full, ro_beam_prev,
    currents_full, currents_prev
):
    """
    Calculate transverse fields as iDST-DCT(mixed_matrix * DST-DCT(RHS.T)).T,
    with and without transposition depending on the field component.
    NOTE: density and currents are assumed to be zero on the perimeter
          (no plasma particles must reach the wall, so the reflection boundary
           must be closer to the center than the simulation window boundary
           minus the coarse plasma particle cloud width).
    """
    jx_prev, jy_prev = currents_prev.jx, currents_prev.jy
    jx_full, jy_full = currents_full.jx, currents_full.jy
    
    ro_half = (currents_full.ro + ro_beam_full +
               currents_prev.ro + ro_beam_prev) / 2
    jz_half = (currents_full.jz + ro_beam_full +
               currents_prev.jz + ro_beam_prev) / 2

    # 0. Calculate gradients and RHS.
    dro_dx, dro_dy = dx_dy(ro_half, grid_step_size)
    djz_dx, djz_dy = dx_dy(jz_half, grid_step_size)
    djx_dxi = (jx_prev - jx_full) / xi_step_size  # - ?
    djy_dxi = (jy_prev - jy_full) / xi_step_size  # - ?

    # We are solving a Helmholtz equation
    Ex_rhs = -(dro_dx - djx_dxi - fields.Ex)  # -?
    Ey_rhs = -(dro_dy - djy_dxi - fields.Ey)
    Bx_rhs = +(djz_dy - djy_dxi + fields.Bx)
    By_rhs = -(djz_dx - djx_dxi - fields.By)

    # Boundary conditions application (for future reference, ours are zero):
    # rhs[:, 0] -= bound_bottom[:] * (2 / grid_step_size)
    # rhs[:, -1] += bound_top[:] * (2 / grid_step_size)

    # 1. Apply our mixed DCT-DST transform to RHS.
    Ey_f = mix2d(Ey_rhs[1:-1, :])[1:-1, :]

    # 2. Multiply f by the magic matrix.
    mix_mat = const.field_mixed_matrix
    Ey_f *= mix_mat

    # 3. Apply our mixed DCT-DST transform again.
    Ey = mix2d(Ey_f)

    # Likewise for other fields:
    Bx = mix2d(mix_mat * mix2d(Bx_rhs[1:-1, :])[1:-1, :])
    By = mix2d(mix_mat * mix2d(By_rhs.T[1:-1, :])[1:-1, :]).T
    Ex = mix2d(mix_mat * mix2d(Ex_rhs.T[1:-1, :])[1:-1, :]).T

    return Ex, Ey, Bx, By


# Pushing particles without any fields (used for initial halfstep estimation) #

def dct2d(a):
    """
    Calculate DCT-Type1-2D, jury-rigged from symmetrically-padded rFFT.
    """
    assert a.shape[0] == a.shape[1]
    N = a.shape[0]
    #                                    //1  2  3  4\ 3  2 \
    # /1  2  3  4\                      | |5  6  7  8| 7  6  |
    # |5  6  7  8|     symmetrically    | |9  A  B  C| B  A  |
    # |9  A  B  C|      padded to       | \D  E  F  G/ F  E  |
    # \D  E  F  G/                      |  9  A  B  C  B  A  |
    #                                    \ 5  6  7  8  7  6 /
    p = cp.zeros((2 * N - 2, 2 * N - 2))
    p[:N, :N] = a
    p[N:, :N] = cp.flipud(a)[1:-1, :]  # flip to right on drawing above
    p[:N, N:] = cp.fliplr(a)[:, 1:-1]  # flip down on drawing above
    p[N:, N:] = cp.flipud(cp.fliplr(a))[1:-1, 1:-1]  # bottom-right corner
    # after padding: rFFT-2D, cut out the top-left segment, take -real part
    return -cp.fft.rfft2(p)[:N, :N].real


def calculate_Bz(grid_step_size, const, currents):
    """
    Calculate Bz as iDCT2D(dirichlet_matrix * DCT2D(djx/dy - djy/dx)).
    """
    # 0. Calculate RHS.
    # NOTE: use gradient instead if available (cupy doesn't have gradient yet).    
    jx, jy = currents.jx, currents.jy

    djx_dy = jx[1:-1, 2:] - jx[1:-1, :-2]
    djy_dx = jy[2:, 1:-1] - jy[:-2, 1:-1]
    djx_dy = cp.pad(djx_dy, 1, 'constant', constant_values=0)
    djy_dx = cp.pad(djy_dx, 1, 'constant', constant_values=0)
    rhs = -(djx_dy - djy_dx) / (grid_step_size * 2)  # -?

    # As usual, the boundary conditions are zero
    # (otherwise add them to boundary cells, divided by grid_step_size/2

    # 1. Apply DST-Type1-2D (Discrete Sine Transform Type 1 2D) to the RHS.
    f = dct2d(rhs)

    # 2. Multiply f by the special matrix that does the job and normalizes.
    f *= const.neumann_matrix

    # 3. Apply iDCT-Type1-2D (Inverse Discrete Cosine Transform Type 1 2D).
    #    We don't have to define a separate iDCT function, because
    #    unnormalized DCT-Type1 is its own inverse, up to a factor 2(N+1)
    #    and we take all scaling matters into account with a single factor
    #    hidden inside neumann_matrix.
    Bz = dct2d(f)

    Bz -= Bz.mean()  # Integral over Bz must be 0.

    return Bz


class FieldComputer(object):
    """
    Class to store some parameters for fields computing.

    Parameters
    ----------
    config : ..config.Config

    Attributes
    ----------
    grid_step_size : float
        Plane grid step size.

    """
    def __init__(self, config: Config):
        self.grid_step_size = config.getfloat('window-width-step-size')
        self.xi_step_size = config.getfloat('xi-step')
        self.trick = config.getfloat('field-solver-subtraction-trick')
        self.variant_A = config.getbool('field-solver-variant-A')

    def compute_fields(
        self, fields, fields_prev, const, rho_beam_full, rho_beam_prev,
        currents_prev, currents_full
    ):
        # Looks terrible! TODO: rewrite this function entirely

        Ex_half, Ey_half, Bx_half, By_half = calculate_Ex_Ey_Bx_By(
            self.grid_step_size, self.xi_step_size, const, fields,
            rho_beam_full, rho_beam_prev, currents_full, currents_prev
        )

        Ex_full = 2 * Ex_half - fields_prev.Ex
        Ey_full = 2 * Ey_half - fields_prev.Ey
        Bx_full = 2 * Bx_half - fields_prev.Bx
        By_full = 2 * By_half - fields_prev.By

        Ez_full = calculate_Ez(self.grid_step_size, const, currents_full)
        Bz_full = calculate_Bz(self.grid_step_size, const, currents_full)
        Phi = calculate_Phi(const, currents_full)

        fields_full = GPUArrays(
            Ex=Ex_full, Ey=Ey_full, Ez=Ez_full,
            Bx=Bx_full, By=By_full, Bz=Bz_full,
            Phi=Phi
        )
        
        Ez_half  = (Ez_full  + fields_prev.Ez) / 2
        Bz_half  = (Bz_full  + fields_prev.Bz) / 2

        fields_half = GPUArrays(
            Ex=Ex_half, Ey=Ey_half, Ez=Ez_half,
            Bx=Bx_half, By=By_half, Bz=Bz_half,
            Phi=Phi
        )

        return fields_full, fields_half
