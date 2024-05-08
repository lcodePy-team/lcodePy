import numpy as np

def _calculate_radial_coefficients(xp, time_midle, profile, plasma_width, 
                                   plasma_width2, plasma_density2, r_init):

    # Uniform radial distribution is default. 
    coef = xp.ones_like(r_init)

    if profile == '1' or profile == 'uniform':
        pass

    elif profile == '2' or profile == 'stepwise': 
        delta_r = plasma_width2 - plasma_width
        if delta_r > 0:
            mask = xp.logical_and(plasma_width < r_init, 
                                  r_init < plasma_width2)
            r_0 = xp.min(r_init[mask])
            coef[mask] = 1 - (r_init[mask] - r_0) / delta_r
        mask = r_init > max(plasma_width, plasma_width2)
        coef[mask] = 0

    elif profile == '3' or profile == 'channel': 
        mask = r_init <= plasma_width
        coef[mask] = plasma_density2
        delta_r = plasma_width2 - plasma_width
        if delta_r > 0:
            mask = xp.logical_and(plasma_width < r_init, 
                                  r_init < plasma_width2)
            r_0 = xp.min(r_init[mask])
            coef[mask] = (plasma_density2 * (1 + delta_r - r_init[mask]) 
                              + (r_init[mask] - r_0)) / delta_r

    elif profile == '4' or profile == 'parabolic-channel':
        coef = plasma_density2 * (1 + (r_init / plasma_width)**2)
        mask = coef > 1
        coef[mask] = 1

    elif profile == '5' or profile == 'gaussian': 
        coef = xp.exp(-r_init**2 / 2 / plasma_width**2)
        mask = r_init > 5 * plasma_width
        coef[mask] = 0

    else:
        print(f'Unknown transverse distribution {profile} at t = {time_midle}, '
              + 'the uniform distribution is used instead.')

    return coef


def _get_rshape_coef(config, time_middle, x_init, y_init):
    available_profiles = {'1', 'uniform', '2', 'stepwise', '3', 'channel',
                          '4', 'parabolic-channel', '5', 'gaussian'}
    xp = config.xp
    plasma_rshape  = config.get('plasma-rshape')
    r_init = xp.sqrt(x_init**2 + y_init**2)
    coef = xp.ones_like(x_init)

    if plasma_rshape in available_profiles:
        plasma_width = config.getfloat('plasma-width')
        plasma_width2 = config.getfloat('plasma-width-2')
        plasma_density2 = config.getfloat('plasma-density-2')
        coef = _calculate_radial_coefficients(xp, time_middle, plasma_rshape, 
                                              plasma_width, 
                                              plasma_width2, 
                                              plasma_density2, r_init)
    else:
        # List comprehension to create an array that is easy to handle
        segments = [line.split() for line in plasma_rshape.splitlines()
                    if len(line.split()) != 0] # without empty lines

        for seg in segments:
            # Check if time_middle lies in one of the regions:
            if time_middle < float(seg[0]):
                profile = seg[1]
                plasma_width = float(seg[2])
                plasma_width2 = float(seg[3])
                plasma_density2 = float(seg[4])
                coef = _calculate_radial_coefficients(xp, time_middle, profile, 
                                                      plasma_width, 
                                                      plasma_width2, 
                                                      plasma_density2, r_init)
                break
            else:
                time_middle -= float(seg[0])
    return coef

def _get_zshape_coef(config, time_middle):
    plasma_zshape  = config.get('plasma-zshape')
    coef = 1
    # Check if plasma_zshape is empty and nothing should be done with q and m
    if not plasma_zshape.isspace():
        # List comprehension to create an array that is easy to handle
        segments = [line.split() for line in plasma_zshape.splitlines()
                    if len(line.split()) != 0] # without empty lines
        # TODO: Do we need to check ValueError?
        for seg in segments:
            # Check if time_middle lies in one of the regions:
            if time_middle < float(seg[0]):
                if seg[2] == 'L': 
                    coef = (float(seg[1]) * (float(seg[0]) - time_middle) 
                            + float(seg[3]) * time_middle) / float(seg[0])
                    break
                else:
                    print(seg[2], 'segment of z shape is not available, '
                          + 'a density of 1 is assumed.')
                    break
            else:
                time_middle -= float(seg[0])
    return coef

def profile_initial_plasma(config, current_time, x_init, y_init, q, m):
    """
    Plasma profiler. 
    """
    xp = config.xp
    # Get required parameters:
    time_step_size = config.getfloat('time-step')
    plasma_shape  = config.getraw('plasma-shape')
    save_zshape   = config.getbool('save-zshape') 
    save_rshape   = config.getbool('save-rshape') 
    time_middle = current_time - time_step_size / 2 # Just like in lcode2d
    # Default value if any other option is not acceptable for this time step.
    coef = 1
    if callable(plasma_shape):
        coef = plasma_shape(current_time, x_init, y_init)
    else: 
        coef_z = _get_zshape_coef(config, time_middle)
        coef_r = _get_rshape_coef(config, time_middle, x_init, y_init)
        coef = coef_z * coef_r

    if xp.any(coef < 0):
        print('A density that is < 0 is not available, ' +
              'a uniform density is assumed.')
        coef = xp.ones_like(x_init)
        mask = xp.ones_like(x_init, dtype=xp.bool_)
    if save_zshape:
        with open('plasma_z_shape.dat', 'a') as f:
            pass
    if save_rshape:
        with open('plasma_r_shape.dat', 'a') as f:
            pass

    # If the total length is too small, hence we finished the loop above,
    # or we broke from the loop, a density of 1 is assumed:
    return q * coef, m * coef

