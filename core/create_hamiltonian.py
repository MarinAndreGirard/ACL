import numpy as np
import qutip as qt
from operators import annihilation_operator

#TODO 
#- will need to changre all functions to the new creat_H_new


def create_H(d1,d2, E_spacing, E_int, E_int2, E_env, E_env2,E_s=0):
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
    #-Go check out the ACL paper, because I may actually be doing an adapted adapted caldereia leggett model.
    #-add seed! usethefollowing
    """
    def get_random_H(n_qubits: int, seed=None) -> np.ndarray:
    np.random.seed(seed)
    mat = np.random.normal(size=(2**n_qubits, 2**n_qubits)) + \
        np.random.normal(size=(2**n_qubits, 2**n_qubits))*1j
    return 1/4*mat*mat.conj().T, None, None
    """

    H_int_s = qt.Qobj(np.zeros([d1,d1]))

    d = d1*d2  # Total Hilbert space dimension
    diagonal_elements = np.arange(0, d1) * E_spacing
    H_s_self = qt.Qobj(np.diag(diagonal_elements))
    ###H_s_self = E_s * qt.rand_herm(d1,1) #TODO add other options/for self interaction of system, use E_Spacing? H_s_self = qt.qeye(d1) or qt.Qobj(np.zeros([d1,d1]))
    H_s = qt.tensor(H_s_self, qt.qeye(d2)) # Extend to full Hilbert space    
       
    ###H_int_s = qt.Qobj(np.diag(diagonal_elements)) # Creat the SHO part of the interaction Hamiltonian
    H_int_e = E_int * qt.rand_herm(d2,1) + E_int2 * qt.qeye(d2) # note that this function rand_herm also takes a density number, which is set at default 1.
    H_int = qt.tensor(H_int_s, H_int_e) # Extend to full Hilbert space

    H_e_self = E_env * qt.rand_herm(d2,1) + E_env2 * qt.qeye(d2)  # Random Hermitian matrix for system 2
    H_e = qt.tensor(qt.qeye(d1), H_e_self) # Extend to full Hilbert space
     
    # Define the total Hamiltonian
    H_total = H_s + H_int + H_e
    
    

    return d, H_total, H_s, H_int, H_e, H_s_self, H_int_s, H_int_e, H_e_self

def create_H_new(d1,d2, E_s, E_s2, E_int_s, E_int_e,E_int_s2,E_int_e2,E_e, E_e2):
    """Creates Hamiltonian for simulation
    Args:
        d1 (int, optional): dimension of system. Defaults to 2.
        d2 (int, optional): dimension of environment. Defaults to 200.
        E_s (float, optional): SHO parameter. Defaults to 1.0.
        E_int (float, optional): interaction strength between the environment and system. Defaults to 0.03.
        E_int2 (int, optional): constant energy factor of interaction. Defaults to 0.
        E_env (int, optional): environment self interaction energy factor. Defaults to 1.
        E_env2 (int, optional): constant energy factor of enviroment self interaction. Defaults to 0.
        
        E_s (int, optional): system self interaction energy factor. Defaults to 0.
        E_s2 (int, optional): constant energy factor of system self interaction. Defaults to 0.

    Returns:
        int, Qobj,Qobj,Qobj...: d, H_total, H_s, H_int, H_e, H_s_self, H_int_s, H_int_e, H_e_self TODO
        The hamiltonian takes 5 5 parameters to define. We also need 2 dimensions for the Hilbert space. And the function also takes other optional parameters, defining its action.
        Note that the random matrices generated cannot be controlled, so the results will be different every time. No random number to control randomness is used by rand_herm.
        System self interaction. Is the self interaction of a truncated simple harmonic oscillator
        System-environment interaction. Is the interaction of a truncated simple harmonic oscillator with a bath
        Environment self interaction. Is the self interaction of a bath
        Parameters: matrix factors: E_s, E_int_s, E_int_e, E_e. Constant factors E_s2, E_int2, E_e2
        The return as a function of parameters is: H = H_s + H_int + H_e = H_s_self x id + H_int_s x H_int_e + id x H_e_self = [(E_s x diag(i) + E_s2 x id)x id ] + [(E_int_s x off_diag_sqrt(i) + E_int_s2 x diag(i)) x (E_int_e x rd_mat1 + E_int_e2 x id)] + [id x (E_e x H_e_self + E_e2 x id)]
    """

    #TODO:
    #-Ability to define more interesting self interaction for system
    #- why is only H_int_s Qobj
    #-finish docstring
    #-Go check out the ACL paper, because I may actually be doing an adapted adapted caldereia leggett model.
    #-add seed! usethefollowing

    #Note that its all defined in the eigenbasis of the SHO. as its self hamiltonian is diagonal here.

    """
    def get_random_H(n_qubits: int, seed=None) -> np.ndarray:
    np.random.seed(seed)
    mat = np.random.normal(size=(2**n_qubits, 2**n_qubits)) + \
        np.random.normal(size=(2**n_qubits, 2**n_qubits))*1j
    return 1/4*mat*mat.conj().T, None, None
    """

    d = d1*d2  # Total Hilbert space dimension

    #Getting the creation and anihilation operators.
    a = annihilation_operator(d1)
    a_dag = a.conj().T

    #making H_s

#    diagonal_elements = np.arange(0, d1) * E_s
#    H_s_self = qt.Qobj(np.diag(diagonal_elements))
    H_s_self = qt.Qobj(np.dot(a,a_dag) * E_s)
    H_s_scale = E_s2*qt.qeye(d1)
    H_s_self= H_s_self+H_s_scale
    H_s = qt.tensor(H_s_self, qt.qeye(d2)) # Extend to full Hilbert space    
    
    #Making H_int_s
#    H_int_s = qt.Qobj(np.zeros([d1,d1]))
#    H_int_s=H_int_s.full()
#    for i in range(d1-1):
#        H_int_s[i,i+1] = np.sqrt(i+1)
#        H_int_s[i+1,i] = np.sqrt(i+1)
#    H_int_s = E_int_s*H_int_s #TODO note that E_int_s and E_s2 are related by the fact that there is a SHO. so at some point i need to change this
#    H_int_s = qt.Qobj(H_int_s)
    H_int_s = qt.Qobj(a+a_dag) * E_int_s
    diagonal_elements = np.arange(0, d1)
    H_int_s_scaleish = E_int_s2*qt.Qobj(np.diag(diagonal_elements))
    H_int_s = H_int_s_scaleish + H_int_s
    
    #Making H_int_e
    H_int_e = E_int_e * qt.rand_herm(d2,1)
    H_int_e_scale = E_int_e2*qt.qeye(d2)
    H_int_e = H_int_e+H_int_e_scale
    #Making H_int
    H_int = qt.tensor(H_int_s, H_int_e) # Extend to full Hilbert space + add constant scale factor

    #Making H_e
    H_e_self = E_e * qt.rand_herm(d2,1) 
    H_e_self_scale = E_e2*qt.qeye(d2)
    H_e_self = H_e_self+H_e_self_scale
    H_e = qt.tensor(qt.qeye(d1), H_e_self) # Extend to full Hilbert space
     
    # Define the total Hamiltonian
    H_total = H_s + H_int + H_e
    
    

    return d, H_total, H_s, H_int, H_e, H_s_self, H_int_s, H_int_e, H_e_self

