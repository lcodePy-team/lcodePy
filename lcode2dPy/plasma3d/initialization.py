"""Module for plasma (3d solver) initialization routines."""
import numpy as np

from ..config.config import Config
from .weights import get_deposit_plasma
from .data import Arrays

ELECTRON_CHARGE = -1
ELECTRON_MASS = 1


# Solving Laplace equation with Dirichlet boundary conditions (Ez) #

def dirichlet_matrix(xp: np, grid_steps, grid_step_size):
    """
    Calculate a magical matrix that solves the Laplace equation
    if you elementwise-multiply the RHS by it "in DST-space".
    See Samarskiy-Nikolaev, p. 187.
    """
    # mul[i, j] = 1 / (lam[i] + lam[j])
    # lam[k] = 4 / h**2 * sin(k * pi * h / (2 * L))**2, where L = h * (N - 1)
    k = xp.arange(1, grid_steps - 1)
    lam = 4 / grid_step_size**2 * xp.sin(k * xp.pi / (2 * (grid_steps - 1)))**2
    lambda_i, lambda_j = lam[:, None], lam[None, :]
    mul = 1 / (lambda_i + lambda_j)
    return mul / (2 * (grid_steps - 1))**2  # additional 2xDST normalization


# Solving Laplace or Helmholtz equation with mixed boundary conditions #

def mixed_matrix(xp: np, grid_steps, grid_step_size, subtraction_coeff):
    """
    Calculate a magical matrix that solves the Helmholtz or Laplace equation
    (subtraction_trick=True and subtraction_trick=False correspondingly)
    if you elementwise-multiply the RHS by it "in DST-DCT-transformed-space".
    See Samarskiy-Nikolaev, p. 189 and around.
    """
    # mul[i, j] = 1 / (lam[i] + lam[j])
    # lam[k] = 4 / h**2 * sin(k * pi * h / (2 * L))**2, where L = h * (N - 1)
    # but k for lam_i spans from 1..N-2, while k for lam_j covers 0..N-1
    ki, kj = xp.arange(1, grid_steps - 1), xp.arange(grid_steps)
    li = 4 / grid_step_size**2 * xp.sin(ki * xp.pi / (2 * (grid_steps - 1)))**2
    lj = 4 / grid_step_size**2 * xp.sin(kj * xp.pi / (2 * (grid_steps - 1)))**2
    lambda_i, lambda_j = li[:, None], lj[None, :]
    mul = 1 / (lambda_i + lambda_j + subtraction_coeff)
    return mul / (2 * (grid_steps - 1))**2  
    # return additional 2xDST normalization


# Solving Laplace equation with Neumann boundary conditions (Bz) #

def neumann_matrix(xp: np, grid_steps, grid_step_size):
    """
    Calculate a magical matrix that solves the Laplace equation
    if you elementwise-multiply the RHS by it "in DST-space".
    See Samarskiy-Nikolaev, p. 187.
    """
    # mul[i, j] = 1 / (lam[i] + lam[j])
    # lam[k] = 4 / h**2 * sin(k * pi * h / (2 * L))**2, where L = h * (N - 1)

    k = xp.arange(0, grid_steps)
    lam = 4 / grid_step_size**2 * xp.sin(k * xp.pi / (2 * (grid_steps - 1)))**2
    lambda_i, lambda_j = lam[:, None], lam[None, :]
    mul = 1 / (lambda_i + lambda_j)  # WARNING: zero division in mul[0, 0]!
    mul[0, 0] = 0  # doesn't matter anyway, just defines constant shift
    return mul / (2 * (grid_steps - 1))**2  # additional 2xDST normalization


# Plasma particles initialization #

def make_plasma_grid(xp: np, steps, step_size, fineness):
    """
    Create initial plasma particles coordinates
    (a single 1D grid for both x and y).
    Avoids positioning particles at the cell edges and boundaries, example:
    `fineness=3`:
        +-----------+-----------+-----------+-----------+
        | .   .   . | .   .   . | .   .   . | .   .   . |
        |           |           |           |           |
        | .   .   . | .   .   . | .   .   . | .   .   . |
        |           |           |           |           |
        | .   .   . | .   .   . | .   .   . | .   .   . |
        +-----------+-----------+-----------+-----------+
    `fineness=2`:
        +-------+-------+-------+-------+-------+
        | .   . | .   . | .   . | .   . | .   . |
        |       |       |       |       |       |
        | .   . | .   . | .   . | .   . | .   . |
        +-------+-------+-------+-------+-------+
    """
    plasma_step = step_size / fineness
    if fineness % 2:  # some on zero axes, none on cell corners
        right_half = xp.arange(steps // 2 * fineness) * plasma_step
        left_half = -right_half[:0:-1]  # invert, reverse, drop zero
    elif 0. < fineness and fineness < 1.:
        right_half = xp.arange(steps // (2 / fineness)) * plasma_step
        left_half = -right_half[:0:-1]  # invert, reverse, drop zero
    else:  # none on zero axes, none on cell corners
        right_half = (.5 + xp.arange(steps // 2 * fineness)) * plasma_step
        left_half = -right_half[::-1]  # invert, reverse
    plasma_grid = xp.concatenate([left_half, right_half])
    assert xp.array_equal(plasma_grid, -plasma_grid[::-1])
    return plasma_grid


def make_plasma_single(xp: np, steps, cell_size, fineness=2):
    """
    Initialize default plasma state, fineness**2 particles per cell.
    """
    pl_grid = make_plasma_grid(xp, steps, cell_size, fineness)

    Np = len(pl_grid)

    y_init = xp.broadcast_to(pl_grid, (Np, Np))
    x_init = y_init.T

    x_offt = xp.zeros((Np, Np))
    y_offt = xp.zeros((Np, Np))
    px = xp.zeros((Np, Np))
    py = xp.zeros((Np, Np))
    pz = xp.zeros((Np, Np))
    q = xp.ones((Np, Np)) * ELECTRON_CHARGE / fineness**2 * cell_size**2
    m = xp.ones((Np, Np)) * ELECTRON_MASS / fineness**2 * cell_size**2

    return x_init, y_init, x_offt, y_offt, px, py, pz, q, m


# Coarse and fine plasma particles initialization #

def make_coarse_plasma_grid(xp: np, steps, step_size, coarseness):
    """
    Create initial coarse plasma particles coordinates
    (a single 1D grid for both x and y).
    """
    assert coarseness == int(coarseness)  # TODO: why?
    plasma_step = step_size * coarseness
    right_half = xp.arange(steps // (coarseness * 2)) * plasma_step
    left_half = -right_half[:0:-1]  # invert, reverse, drop zero
    plasma_grid = xp.concatenate([left_half, right_half])
    assert(xp.array_equal(plasma_grid, -plasma_grid[::-1]))
    return plasma_grid


def make_fine_plasma_grid(xp: np, steps, step_size, fineness):
    """
    Create initial fine plasma particles coordinates
    (a single 1D grid for both x and y).
    Avoids positioning particles at the cell edges and boundaries, example:
    `fineness=3` (and `coarseness=2`):
        +-----------+-----------+-----------+-----------+
        | .   .   . | .   .   . | .   .   . | .   .   . |
        |           |           |           |           |   . - fine particle
        | .   .   . | .   *   . | .   .   . | .   *   . |
        |           |           |           |           |   * - coarse particle
        | .   .   . | .   .   . | .   .   . | .   .   . |
        +-----------+-----------+-----------+-----------+
    `fineness=2` (and `coarseness=2`):
        +-------+-------+-------+-------+-------+
        | .   . | .   . | .   . | .   . | .   . |           . - fine particle
        |       |   *   |       |   *   |       |
        | .   . | .   . | .   . | .   . | .   . |           * - coarse particle
        +-------+-------+-------+-------+-------+
    """
    assert fineness == int(fineness)
    plasma_step = step_size / fineness
    if fineness % 2:  # some on zero axes, none on cell corners
        right_half = xp.arange(steps // 2 * fineness) * plasma_step
        left_half = -right_half[:0:-1]  # invert, reverse, drop zero
    else:  # none on zero axes, none on cell corners
        right_half = (.5 + xp.arange(steps // 2 * fineness)) * plasma_step
        left_half = -right_half[::-1]  # invert, reverse
    plasma_grid = xp.concatenate([left_half, right_half])
    assert xp.array_equal(plasma_grid, -plasma_grid[::-1])
    return plasma_grid


def make_plasma_dual(xp: np, steps, cell_size, coarseness=2, fineness=2):
    """
    Make coarse plasma initial state arrays and the arrays needed to intepolate
    coarse plasma into fine plasma (`virt_params`).
    Coarse is the one that will evolve and fine is the one to be bilinearly
    interpolated from the coarse one based on the initial positions
    (using 1 to 4 coarse plasma particles that initially were the closest).
    """
    coarse_step = cell_size * coarseness

    # Make two initial grids of plasma particles, coarse and fine.
    # Coarse is the one that will evolve and fine is the one to be bilinearly
    # interpolated from the coarse one based on the initial positions.

    coarse_grid = make_coarse_plasma_grid(xp, steps, cell_size, coarseness)
    coarse_grid_xs, coarse_grid_ys = coarse_grid[:, None], coarse_grid[None, :]

    fine_grid = make_fine_plasma_grid(xp, steps, cell_size, fineness)
    fine_grid_xs, fine_grid_ys = fine_grid[:, None], fine_grid[None, :]

    Nc = len(coarse_grid)

    # Create plasma electrons on the coarse grid, the ones that really move
    coarse_x_init = xp.broadcast_to(xp.asarray(coarse_grid_xs), (Nc, Nc))
    coarse_y_init = xp.broadcast_to(xp.asarray(coarse_grid_ys), (Nc, Nc))
    coarse_x_offt = xp.zeros((Nc, Nc))
    coarse_y_offt = xp.zeros((Nc, Nc))
    coarse_px = xp.zeros((Nc, Nc))
    coarse_py = xp.zeros((Nc, Nc))
    coarse_pz = xp.zeros((Nc, Nc))
    coarse_m = xp.ones((Nc, Nc)) * ELECTRON_MASS * coarseness**2 * cell_size**2
    coarse_q = xp.ones((Nc, Nc)) * ELECTRON_CHARGE * coarseness**2 * cell_size**2

    # Calculate indices for coarse -> fine bilinear interpolation

    # Neighbour indices array, 1D, same in both x and y direction.
    indices = xp.searchsorted(coarse_grid, fine_grid)
    # example:
    #     coarse:  [-2., -1.,  0.,  1.,  2.]
    #     fine:    [-2.4, -1.8, -1.2, -0.6,  0. ,  0.6,  1.2,  1.8,  2.4]
    #     indices: [ 0  ,  1  ,  1  ,  2  ,  2  ,  3  ,  4  ,  4  ,  5 ]
    # There is no coarse particle with index 5, so clip it to 4:
    indices_next = xp.clip(indices, 0, Nc - 1)  # [0, 1, 1, 2, 2, 3, 4, 4, 4]
    # Clip to zero for indices of prev particles as well:
    indices_prev = xp.clip(indices - 1, 0, Nc - 1)  # [0, 0, 0, 1 ... 3, 3, 4]
    # mixed from: [ 0&0 , 0&1 , 0&1 , 1&2 , 1&2 , 2&3 , 3&4 , 3&4, 4&4 ]

    # Calculate weights for coarse->fine interpolation from initial positions.
    # The further the fine particle is from closest right coarse particles,
    # the more influence the left ones have.
    influence_prev = (coarse_grid[indices_next] - fine_grid) / coarse_step
    influence_next = (fine_grid - coarse_grid[indices_prev]) / coarse_step
    # Fix for boundary cases of missing cornering particles.
    influence_prev[indices_next == 0] = 0   # nothing on the left?
    influence_next[indices_next == 0] = 1   # use right
    influence_next[indices_prev == Nc - 1] = 0  # nothing on the right?
    influence_prev[indices_prev == Nc - 1] = 1  # use left
    # Same arrays are used for interpolating in y-direction.

    # The virtualization formula is thus
    # influence_prev[pi] * influence_prev[pj] * <bottom-left neighbour value> +
    # influence_prev[pi] * influence_next[nj] * <top-left neighbour value> +
    # influence_next[ni] * influence_prev[pj] * <bottom-right neighbour val> +
    # influence_next[ni] * influence_next[nj] * <top-right neighbour value>
    # where pi, pj are indices_prev[i], indices_prev[j],
    #       ni, nj are indices_next[i], indices_next[j] and
    #       i, j are indices of fine virtual particles

    # This is what is employed inside mix() and deposit_kernel().

    # An equivalent formula would be
    # inf_prev[pi] * (inf_prev[pj] * <bot-left> + inf_next[nj] * <bot-right>) +
    # inf_next[ni] * (inf_prev[pj] * <top-left> + inf_next[nj] * <top-right>)

    # Values of m, q, px, py, pz should be scaled by 1/(fineness*coarseness)**2

    virt_params = Arrays(xp,
        influence_prev=influence_prev, influence_next=influence_next,
        indices_prev=indices_prev, indices_next=indices_next,
        fine_x_init = fine_grid_xs, fine_y_init = fine_grid_ys
    )

    return (
        coarse_x_init, coarse_y_init, coarse_x_offt, coarse_y_offt,
        coarse_px, coarse_py, coarse_pz, coarse_q, coarse_m, virt_params
    )


# TODO: when more zshapes are created, move this fun to a separate file.
def change_by_plasma_zshape(config: Config, current_time, q, m):
    # Get required parameters:
    time_step_size = config.getfloat('time-step')
    plasma_zshape  = config.get('plasma-zshape')
    time_middle = current_time - time_step_size / 2 # Just like in lcode2d

    # Check if plasma_zshape is empty and nothing should be done with q and m
    if plasma_zshape.isspace():
        return q, m

    # List comprehension to create an array that is easy to handle
    lines = [line.split() for line in plasma_zshape.splitlines()
             if len(line) != 0] # without empty lines

    # TODO: Do we need to check ValueError?
    for line in lines:
        # Check if time_middle lies in one of the regions:
        if time_middle < float(line[0]):
            if line[2] == 'L': # Only linear rise/decrease is implemented now
                coef = (float(line[1]) * (float(line[0]) - time_middle) +
                        float(line[3]) * time_middle) / float(line[0])
                if coef >= 0:
                    return q * coef, m * coef
                else:
                    print('A density that is < 0 is not available, ' +
                          'a density of 1 is assumed.')
                    break
            else:
                print(line[2], 'segment shape is not available, a density ' +
                      'of 1 is assumed.')
                break
        else:
            time_middle -= float(line[0])

    # If the total length is too small, hence we finished the loop above,
    # or we broke from the loop, a density of 1 is assumed:
    return q, m


def init_plasma(config: Config, current_time=0):
    """
    Initialize all the arrays needed (for what?).
    """
    pu_type = config.get('processing-unit-type').lower()
    if pu_type == 'cpu':
        xp = np
    elif pu_type == 'gpu':
        import cupy as xp

    grid_steps            = config.getint('window-width-steps')
    grid_step_size        = config.getfloat('window-width-step-size')
    reflect_padding_steps = config.getint('reflect-padding-steps')
    plasma_padding_steps  = config.getint('plasma-padding-steps')
    plasma_fineness       = config.getfloat('plasma-fineness')
    dual_plasma_approach  = config.getbool('dual-plasma-approach')
    subtraction_coeff = config.getfloat('field-solver-subtraction-coefficient')
    xi_step_size = config.getfloat('xi-step')

    if plasma_fineness > 1:
        plasma_fineness = int(plasma_fineness)

    # for convenient diagnostics, a cell should be in the center of the grid
    assert grid_steps % 2 == 1

    # particles should not reach the window pre-boundary cells
    if reflect_padding_steps <= 2:
        raise Exception("'reflect_padding_steps' parameter is too low.\n" +
                        "Details: 'reflect_padding_steps' must be bigger than" +
                        " 2. By default it is 5.")

    if dual_plasma_approach and pu_type == 'gpu':
        plasma_coarseness = config.getint('plasma-coarseness')

        # virtual particles should not reach the window pre-boundary cells
        assert reflect_padding_steps > plasma_coarseness + 1
        # TODO: The (costly) alternative is to reflect after plasma virtualization,
        #       but it's better for stabitily, or is it?

        x_init, y_init, x_offt, y_offt, px, py, pz, q, m, virt_params = \
            make_plasma_dual(
                xp, steps=grid_steps - 2*plasma_padding_steps,
                cell_size=grid_step_size, coarseness=plasma_coarseness,
                fineness=plasma_fineness)

    elif dual_plasma_approach and pu_type == 'cpu':
        Exception("We cannot use this type of processing unit." +
                  "Please choose GPU if you want to use dual plasma approach.")
    else:
        plasma_coarseness, virt_params = None, Arrays(xp)

        x_init, y_init, x_offt, y_offt, px, py, pz, q, m = \
            make_plasma_single(
                xp, steps=grid_steps - plasma_padding_steps * 2,
                cell_size=grid_step_size, fineness=plasma_fineness)

    # Here we change q and m arrays of plasma particles according to
    # set plasma_zhape:
    q, m = change_by_plasma_zshape(config, current_time, q, m)

    # We create arrays dx_chaotic, dy_chaotic, dx_chaotic_perp, dy_chaotic_perp
    # that are used in noise filter, with right sizes:
    size = x_offt.shape[0]
    dx_chaotic = xp.zeros((size, size), dtype=xp.float64)
    dy_chaotic = xp.zeros((size, size), dtype=xp.float64)
    dx_chaotic_perp = xp.zeros((size, size), dtype=xp.float64)
    dy_chaotic_perp = xp.zeros((size, size), dtype=xp.float64)

    particles = Arrays(xp, x_init=x_init, y_init=y_init,
                       x_offt=x_offt, y_offt=y_offt, px=px, py=py, pz=pz,
                       q=q, m=m,
                       dx_chaotic=dx_chaotic,
                       dy_chaotic=dy_chaotic,
                       dx_chaotic_perp=dx_chaotic_perp, 
                       dy_chaotic_perp=dy_chaotic_perp)

    # Determine the background ion charge density by depositing the electrons
    # with their initial parameters and negating the result.
    initial_deposition = get_deposit_plasma(config)

    ro_electrons_initial, _, _, _ = initial_deposition(particles, virt_params)
    ro_initial = -ro_electrons_initial

    dir_matrix = dirichlet_matrix(xp, grid_steps, grid_step_size)
    mix_matrix = mixed_matrix(xp, grid_steps, grid_step_size, subtraction_coeff)
    neu_matrix = neumann_matrix(xp, grid_steps, grid_step_size)

    grid = ((np.arange(grid_steps) - grid_steps // 2) * grid_step_size)

    if dual_plasma_approach:
        const_arrays = Arrays(xp,
            grid=grid,
            ro_initial=ro_initial, dirichlet_matrix=dir_matrix, 
            field_mixed_matrix=mix_matrix, neumann_matrix=neu_matrix,
            influence_prev=virt_params.influence_prev,
            influence_next=virt_params.influence_next,
            indices_prev=virt_params.indices_prev,
            indices_next=virt_params.indices_next,
            fine_x_init=virt_params.fine_x_init,
            fine_y_init=virt_params.fine_y_init)
    else:
        const_arrays = Arrays(xp,
            grid=grid,
            ro_initial=ro_initial, dirichlet_matrix=dir_matrix,
            field_mixed_matrix=mix_matrix,
            neumann_matrix=neu_matrix)

    def zeros():
        return xp.zeros((grid_steps, grid_steps), dtype=xp.float64)

    fields = Arrays(xp, Ex=zeros(), Ey=zeros(), Ez=zeros(),
                    Bx=zeros(), By=zeros(), Bz=zeros(), Phi=zeros())
    currents = Arrays(xp, ro=zeros(), jx=zeros(), jy=zeros(), jz=zeros())
    
    xi_plasma_layer = xi_step_size # We need it this way to start from xi_i = 0

    return fields, particles, currents, const_arrays, xi_plasma_layer


def load_plasma(config: Config, path_to_plasmastate: str):
    fields, particles, currents, const_arrays, xi_plasma_layer =\
        init_plasma(config)

    with fields.xp.load(file=path_to_plasmastate) as state:
        fields = Arrays(fields.xp,
                        Ex=state['Ex'], Ey=state['Ey'], Ez=state['Ez'],
                        Bx=state['Bx'], By=state['By'], Bz=state['Bz'],
                        Phi=state['Phi'])

        particles = Arrays(fields.xp,
                           x_init=state['x_init'], y_init=state['y_init'],
                           q=state['q'], m=state['m'],
                           x_offt=state['x_offt'], y_offt=state['y_offt'],
                           px=state['px'], py=state['py'], pz=state['pz'], 
        
                           dx_chaotic=state['dx_chaotic'],
                           dy_chaotic=state['dy_chaotic'],
                           dx_chaotic_perp=state['dx_chaotic_perp'],
                           dy_chaotic_perp=state['dy_chaotic_perp'])

        currents = Arrays(fields.xp,
                          ro=state['ro'], jx=state['jx'],
                          jy=state['jy'], jz=state['jz'])

        try:
            const_arrays.ro_initial = state['ro_initial']
        except:
            pass
        
        xi_plasma_layer = state['xi_plasma_layer'][0]

    return fields, particles, currents, const_arrays, xi_plasma_layer
