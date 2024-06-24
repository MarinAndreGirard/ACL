import qutip as qt
import numpy as np

def annihilation_operator(dimension):
    """Creates the annihilation operator for a given dimension

    Args:
        dimension (int): dimension of Hilbert space

    Returns:
        Qobj: annihilation operator
    """
    a=qt.Qobj(np.zeros([dimension,dimension]))
    a=a.full()
    for i in range(dimension-1):
        a[i,i+1] = np.sqrt(i+1)
    return a

