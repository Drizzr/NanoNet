
import numpy as np

def vectorized_result(j, length):
    """
    Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network.
    """
    e = np.zeros(length)
    e[j] = 1.0
    return e