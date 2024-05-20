import numpy as np
import qutip as qt
import math
import matplotlib.pyplot as plt
import create_hamiltonian as ch
import create_state as cs

#TODO make sure I have all the cool features of q_solve before closing it forever

def time_evo(d1=10,d2=200,E_spacing = 1.0, E_int = 0.03, E_int2=0, E_env=1, E_env2=0,w=[0], tmax= 10, ind_nb = 100,log=0):

    H_list = ch.create_H(d1,d2,E_spacing, E_int, E_int2, E_env, E_env2)
    H=H_list[1]
    state_list = cs.create_state(d1,d2,H_list[8],w) 
    
    tlist = np.linspace(0, tmax, ind_nb) # Linear spacing
    if log == 0:
        tlist = np.linspace(0, tmax, ind_nb)  # Linear spacing
    elif log == 1:
        tlist = np.logspace(np.log10(1), np.log10(tmax+1), ind_nb)-1  # Logarithmic spacing
    else:
        raise ValueError("Invalid value for 'log'. It should be either 0 or 1.")
    
    # Perform time evolution of the combined system
    result = qt.mesolve(H, state_list[0], tlist, [], []) #TODO what are the other things this mesolve takes? is mesolve the right one to use?
    
    return result, tlist, H_list, state_list