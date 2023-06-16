import numpy as np
import tensorflow as tf


def gaussian(shape, unit, radius):
    """
    a function that receives the shape of a matrix and generates a matrix with gaussian distribution
    when guided in a fiber, the electric field of the light is gaussian and the phase is evenly distributed
    Args:
        shape: matrix shape (x,y)
        unit: the distance between each point in the matrix
        radius: the radius of the gaussian distribution

    Returns:
        a matrix with gaussian distribution
    """
    x_coord, y_coord = np.indices(shape, sparse=True)
    r_square = unit ** 2 * (x_coord - (shape[0] - 1) / 2) ** 2 + unit ** 2 * (y_coord - (shape[1] - 1) / 2) ** 2
    # print(r_square)
    profile = np.exp(-2 * r_square / (radius ** 2))
    norm_profile = profile / np.sum(profile)  # we want the energy to be computed is 1
    return np.sqrt(norm_profile)


def arg_phase_to_complex(arg, phase):
    """
    a function that receives the argument and phase of a complex number and returns the complex number
    Args:
        arg: the argument of the complex number
        phase: the phase of the complex number

    Returns:
        the complex number
    """
    cs = tf.math.cos(phase)
    si = tf.math.sin(phase)
    return tf.complex(arg * cs, arg * si)


def cross_integral(x, y):
    """
    a function that receives two matrices and calculates the cross integral between them
    Args:
        x: matrix 1
        y: matrix 2
        unit: the distance between each point in the matrix

    Returns:
        the cross integral between the two matrices
    """
    return tf.abs(tf.reduce_sum(tf.math.conj(x) * y))
