import numpy as np


def loc_offset_to_scaled_value(v_g, v_d, d_s):
    """Offset training value for location distances.
    Idea d_x + z = x_g
    But learn 'z/d_s' instead of z directly
    Parameters
    ----------
    v_g : float
        location ground truth, either x_g or y_g
    v_d : float
        location of default box eiter d_x or d_y
    d_s : float
        default box size, either h or w

    Returns
    -------
    float
        l: scaled value to be learned by the network
    """
    # v_d + z = v_g
    z = (v_g - v_d)
    return z/d_s


def add_loc_offset(v_d, l, d_s):
    """Compute the real location from the default boxes location v_d and
    the scaled offset value l, learned by the network.

    Parameters
    ----------
    v_d : float
        location of the default box, either v_x or v_y
    l : float
        scaled offset value as computed by the network
    d_s : float
        default box size value, either v_w or v_h

    Returns
    -------
    float
        v_g, (if the network is correct)
    """
    return v_d + l*d_s


def size_factor_to_scaled_value(u_g, d_s):
    """Compute the scaled value of a box size that the network should learn

    Parameters
    ----------
    u_g : float
        ground truth box size, either h_g or e_g
    d_s : float
        default box size, either d_h or d_w

    Returns
    -------
    float
        scaled value l.
    """
    if u_g == 0:
        return 1e-9
    return np.log(u_g/d_s)


def mul_size_factor(d_s, l):
    """Compute the real size of a given default box size d_s and the 
    scaled factor l as computed by the network.

    Parameters
    ----------
    d_s : [type]
        [description]
    l : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return np.exp(l)*d_s
