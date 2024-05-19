import numpy as np
import qutip as qt
import math
import matplotlib.pyplot as plt
import create_hamiltonian as ch
import create_state as cs

#TODO make sure I have all the cool features of q_solve before closing it forever

def time_evo(tmax= 10, ind_nb = 100,log=0):

    H_list = ch.create_H(2,200,1.0,0.03,0,1,0)
    H=H_list[1]
    state_list = cs.create_state(2,200,H_list[8]) 
    
    tlist = np.linspace(0, tmax, ind_nb) # Linear spacing
    if log == 0:
        tlist = np.linspace(0, tmax, ind_nb)  # Linear spacing
    elif log == 1:
        tlist = np.logspace(numpy.log10(1), numpy.log10(tmax+1), ind_nb)-1  # Logarithmic spacing
    else:
        raise ValueError("Invalid value for 'log'. It should be either 0 or 1.")
    
    # Perform time evolution of the combined system
    result = qt.mesolve(H, state, tlist, [], []) #TODO what are the other things this mesolve takes? is mesolve the right one to use?
    
    return result, tlist, H_list, ket_list