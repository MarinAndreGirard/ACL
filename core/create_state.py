import numpy as np
import qutip as qt
import math
from operators import annihilation_operator
from scipy.linalg import expm

def create_state(d1,d2, H_e_self,w,envi=[0]):
    """_summary_

    Args:
        d1 (int): _description_
        d2 (int): _description_
        H_e_self (Qobj): Environment self interaction Hamiltonian
        w (array): array of weights for the initial state of the system in the SHO energy eigenstates
        envi (array, optional): array of weights for the environment state in the environment self interaction eigenstates. Defaults to [0] which sets it to the d2/2 eigenstate.

    Raises:
        ValueError: if envi is not either [0] or of length d2

    Returns:
        _type_: _description_ TODO
    """
        
    #TODO:
    #-Find a way to use a randome key to have consistent results and re-use them for testing.
    #-Add option to strt in non product state
    #-Add option of starting in not an environment self interaction eigenstate (or superposition of them)
    #-finish docstring

    ket_list = [qt.basis(d1, i) for i in range(d1)] #Define the basis states of the system

    state_s = sum([w[i]*ket_list[i] for i in range(len(ket_list))]).unit() #Define the initial state of the system

    ev ,es = H_e_self.eigenstates()
    if np.array_equal(envi, [0]):
        state_e = es[round(d2/2)] #Define initial state of environment case 1
    else:
        l = len(envi)
        if l != d2:
            raise ValueError("Length of 'envi' and 'd2' must be the same")
        state_e = sum([envi[i]*es[i] for i in range(len(ket_list))]).unit() #Define initial state of environment case 2

    #define initial state of full system
    state = qt.tensor(state_s, state_e)

    return state, ket_list

def create_coherent_state(d1,alpha=1j):
    a=annihilation_operator(d1)
    a_dag = a.conj().T
    a_qobj = qt.Qobj(a)
    
    eig_ener_a,eig_states_a=a_qobj.eigenstates()
    unic_eig=eig_states_a[0].full()

    C=expm(alpha*a-alpha.conjugate()*a_dag)
    coherent_state = qt.Qobj(np.dot(C,unic_eig))
    return coherent_state
