import numpy as np
import qutip as qt

def create_H(d1,d2, E_spacing, E_int, E_int2, E_env, E_env2):
    """Creates Hamiltonian for simulation

    Args:
        d1 (int, optional): dimension of system. Defaults to 2.
        d2 (int, optional): dimension of environment. Defaults to 200.
        E_spacing (float, optional): SHO parameter. Defaults to 1.0.
        E_int (float, optional): interaction strength between the environment and system. Defaults to 0.03.
        E_int2 (int, optional): constant energy factor of interaction. Defaults to 0.
        E_env (int, optional): environment self interaction energy factor. Defaults to 1.
        E_env2 (int, optional): constant energy factor of enviroment self interaction. Defaults to 0.

    Returns:
        int, Qobj,Qobj,Qobj...: d, H_total, H_s, H_int, H_e, H_s_self, H_int_s, H_int_e, H_e_self TODO
        The hamiltonian takes 5 5 parameters to define. We also need 2 dimensions for the Hilbert space. And the function also takes other optional parameters, defining its action.
        Note that the random matrices generated cannot be controlled, so the results will be different every time. No random number to control randomness is used by rand_herm.
        System self interaction. Is the self interaction of a truncated simple harmonic oscillator
        System-environment interaction. Is the interaction of a truncated simple harmonic oscillator with a bath
        Environment self interaction. Is the self interaction of a bath
    """

    #TODO:
    #-Ability to define more interesting self interaction for system
    #- why is only H_int_s Qobj
    #-finish docstring
    #-

    d = d1*d2  # Total Hilbert space dimension
        
    H_s_self = 0.01 * qt.rand_herm(d1,1) #TODO add other options/for self interaction of system, use E_Spacing? H_s_self = qt.qeye(d1) or qt.Qobj(np.zeros([d1,d1]))
    H_s = qt.tensor(H_s_self, qt.qeye(d2)) # Extend to full Hilbert space    
    
    diagonal_elements = np.arange(0, d1) * E_spacing   
    H_int_s = qt.Qobj(np.diag(diagonal_elements)) # Creat the SHO part of the interaction Hamiltonian
    H_int_e = E_int * qt.rand_herm(d2,1) + E_int2 * qt.qeye(d2) # note that this function rand_herm also takes a density number, which is set at default 1.
    H_int = qt.tensor(H_int_s, H_int_e) # Extend to full Hilbert space

    H_e_self = E_env * qt.rand_herm(d2,1) + E_env2 * qt.qeye(d2)  # Random Hermitian matrix for system 2
    H_e = qt.tensor(qt.qeye(d1), H_e_self) # Extend to full Hilbert space
     
    # Define the total Hamiltonian
    H_total = H_s + H_int + H_e
    
    

    return d, H_total, H_s, H_int, H_e, H_s_self, H_int_s, H_int_e, H_e_self
