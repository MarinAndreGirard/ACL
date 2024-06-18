import numpy as np
import qutip as qt
import math
import matplotlib.pyplot as plt

#This is a set of basic functions that output basic quantities useful fo rplotting the main quantities of interest

def state_distribution_eig_tot(result, eig, info_list):
    """
    Calculate the state probability distribution based on the total eigenvalues.

    Parameters:
        result (object): The result object containing the states.
        eig (tuple): A tuple containing the total eigenvalues.
        info_list (list): A list containing additional information.

    Returns:
        tuple: A tuple containing the total eigenvalues and the energy coefficients.
    """

    eigenenergies_total = eig[0]
    eigenstates_total = eig[1]
    EI = info_list[3]
    state = result.states[0]
    energy_coeff=[abs(np.vdot(state, eigenstate)) ** 2 for eigenstate in eigenstates_total]
    
    return eigenenergies_total,energy_coeff

def schmidt_distrib_eig_tot(result,eig,tlist):
    #TODO need to make it a function that gives back the distribution for schmidt states in all Hamiltonians. Or for select Hamiltonians.
    """
    Calculate the Schmidt state probability distribution based on the total eigenvalues.

    Args:
        result (object): The result object containing the states.
        eig (tuple): A tuple containing the total eigenvalues.
        tlist (list): A list of time points.
        info_list (list): A list containing additional information.

    Returns:
        tuple: A tuple containing two lists of energy coefficients for each Schmidt state.
    """
    prob_list1 = []
    prob_list2 = []
    eigenenergies_total = eig[0]
    eigenstates_total = eig[1]
    for idx in range(len(tlist)-1):
        #TODO use new function once you have a working compute_schmidt_full_new function
        state = compute_schmidt_full(result,idx+1,1)
        state2 = compute_schmidt_full(result,idx+1,2)
        energy_coeff=[abs(np.vdot(state, eigenstate)) ** 2 for eigenstate in eigenstates_total]
        energy_coeff2=[abs(np.vdot(state2, eigenstate)) ** 2 for eigenstate in eigenstates_total]
        prob_list1.append(energy_coeff)
        prob_list2.append(energy_coeff2)
    return prob_list1, prob_list2, eigenenergies_total


def Neff(result,eig):
    """
    Calculate the quantity (Neff) based on the given result and eigenvalues. #TODO explain Neff

    Parameters:
    - result (object): The result object containing the states.
    - eig (tuple): A tuple containing the total eigenvalues.

    Returns:
    - Neff (float): TODO
    - eigenenergies_total (list): The list of total eigenvalues.
    """
    
    eigenenergies_total = eig[0]
    eigenstates_total = eig[1]
    state = result.states[0]
    p2=[(abs(np.vdot(state, eigenstate)) ** 2) ** 2 for eigenstate in eigenstates_total]
    Neff = 1/np.sum(p2)

    return Neff,eigenenergies_total


def random_dephasing(state):
    dephased_data = [elem * np.exp(1j * np.random.uniform(0, 2*np.pi)) for elem in state.full().ravel()]
    dephased_state = qt.Qobj(np.array(dephased_data).reshape(state.shape))
    return dephased_state


"""
Side project. this piece of code dephases a state and calculates its expectation values. the goal of which was to compare them to the equilibrium values.
It already works. but i may have needed to only dephase the environment part. unsure.

st=result.states[50]
depahsed_st=random_dephasing(st)
print(st)

E_s_dephased= np.inner(np.conj(depahsed_st).T, np.matmul(H_list[2].full(),depahsed_st))
E_int_dephased= np.inner(np.conj(depahsed_st).T, np.matmul(H_list[3].full(),depahsed_st))
E_e_dephased= np.inner(np.conj(depahsed_st).T, np.matmul(H_list[4].full(),depahsed_st))

exp_val_time_dephased = [np.abs(E_s_dephased), np.abs(E_int_dephased), np.abs(E_e_dephased)]        
print(exp_val_time_dephased)


"""
