import numpy as np
import qutip as qt

#H takes 5 parameters, + 2 dimensions + option selecting
#E_spacing for qs, E_int interaction factor, E_int2 constant added to env part of interaction term
# E_env env self interaction factor, E_env2 constant part of env self interaction term
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
        int, Qobj,Qobj,Qobj...: _description_...TODO
    """
    d = d1*d2  # Total Hilbert space dimension

    # System self interaction. Is the self interaction of a truncated simple harmonic oscillator
    # We also want the option to make it a random matrix or the identity.
    # or an identity matrix with simple spacing (which might just be my SHO anyways) TODO
    
    H_s_self = qt.qeye(d1) #TODO add other options/for self interaction of system, use E_Spacing?
    H_s = qt.tensor(H_s_self, qt.qeye(d2)) # Makes the full system Hamiltonian
    
    # System-environment interaction. Is the interaction of a truncated simple harmonic oscillator with a bath
    
    diagonal_elements = np.arange(0, d1) * E_spacing   
    H_int_s = qt.Qobj(np.diag(diagonal_elements)) # Creat the SHO part of the interaction Hamiltonian TODO is that what it is supposed to be?
    H_int_e = E_int *qt.rand_herm(d2,1) + E_int2 * qt.qeye(d2)
    H_int = qt.tensor(H_int_s, H_int_e) # Makes the full interaction Hamiltonian
    
    # Environment self interaction. Is the self interaction of a bath
    H_e_self = E_env * qt.rand_herm(d2,1) + E_env2 * qt.qeye(d2)  # Random Hermitian matrix for system 2
    H_e = qt.tensor(qt.qeye(d1), H_e_self) # Makes the full environment Hamiltonian
    
    # Define the total Hamiltonian
    H_total = H_s + H_int + H_e
    
    #TODO why is only H_int_s Qobj

    return d, H_total, H_s, H_int, H_e, H_s_self, H_int_s, H_int_e, H_e_self
