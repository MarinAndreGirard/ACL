import numpy as np
import qutip as qt
import math
import matplotlib.pyplot as plt
import create_hamiltonian as ch
import create_state as cs

#TODO make sure I have all the cool features of q_solve before closing it forever

def time_evo(d1=10,d2=200,E_spacing = 1.0, E_int = 0.03, E_int2=0, E_env=1, E_env2=0,w=[0,0,0,np.sqrt(0.3),0,0,0,np.sqrt(0.7),0,0],envi=[0], tmax= 10, ind_nb = 100,log=0):
    """_summary_

    Args:
        d1 (int, optional): dimension of the system. Defaults to 10.
        d2 (int, optional): dimension of the environment. Defaults to 200.
        E_spacing (float, optional): energy between each level of the truncated simple harmonic oscillator. Defaults to 1.0.
        E_int (float, optional): interaction strength between the system and enviroment. Defaults to 0.03.
        E_int2 (int, optional): constant energy of the interaction term. Defaults to 0.
        E_env (int, optional): energy factor in front of environment self hamiltonian. Defaults to 1.
        E_env2 (int, optional): constant enerfy of the environemnt self interaction. Defaults to 0.
        w (list, optional): list of probabilities of the initial state of the system to be in the SHO energy eigenstates. Defaults to [0,0,0,np.sqrt(0.3),0,0,0,np.sqrt(0.7),0,0].
        tmax (int, optional): max time of the time evolution. Defaults to 10.
        ind_nb (int, optional): number of time steps. Defaults to 100.
        log (int, optional): defines if steps are taken linearly (0) or logarithmically (1). Defaults to 0.

    Raises:
        ValueError: you need as many weights as there are dimensions in the system

    Returns:
        result, tlist, H_list, state_list: result (TODO), state_list (TODO), H_list (TODO), tlist: list of times at which the time evolution was calculated
    """
    if len(w) != d1:
        raise ValueError("Length of 'w' and 'd1' must be the same")

    H_list = ch.create_H(d1,d2,E_spacing, E_int, E_int2, E_env, E_env2)
    H=H_list[1]
    state_list = cs.create_state(d1,d2,H_list[8],w,envi) 
     
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