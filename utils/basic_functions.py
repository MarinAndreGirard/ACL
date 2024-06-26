import numpy as np
import qutip as qt
import math
import matplotlib.pyplot as plt
from scipy.linalg import expm


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

def random_dephasing_energy_basis(state,H_list):
    #takes state in b basis, expresses it in energy basis dephases it, re-expresses it in the b basis and returns it.
    d=H_list[0]
    H=H_list[1]
    eigenenergies,eigenstates=H.eigenstates()
    state_in_energy=[]
    for i in range(d):
        w=eigenstates[i].dag()*state
        w=w*np.exp(1j * np.random.uniform(0, 2*np.pi))
        state_in_energy.append(w.full().ravel())
    state_in_energy = np.array(state_in_energy)
    #for i in range(d):
    #    ...*state_in_energy
    #big problem of tryi ng to change the basis back.
    return state_in_energy

def random_dephasing(state):
    dephased_data = [elem * np.exp(1j * np.random.uniform(0, 2*np.pi)) for elem in state.full().ravel()]
    dephased_state = qt.Qobj(np.array(dephased_data).reshape(state.shape))
    return dephased_state

def get_state_probabilities(result, eigenstates, time_index, sys_env=0):

    state = result.states[time_index]
    if sys_env == 0:
        density_matrix = qt.ptrace(state, [0]) # density matrix of the system
    elif sys_env == 1:
        density_matrix = qt.ptrace(state, [1]) # density matrix of the environment

    #define a check for the denisty matrix and tyhe eigenstates being the same dimenisons
    p = [qt.expect(density_matrix,eigenstate) for eigenstate in eigenstates]
    return p

"""
Side project. this piece of code dephases a state and calculates its expectation values. the goal of which was to compare them to the equilibrium values.
It already works. but i may have needed to only dephase the environment part. unsure.


def random_dephasing(state):
    state = state.full().ravel()
    dephased_data = [elem * np.exp(1j * np.random.uniform(0, 2*np.pi)) for elem in state]
    dephased_state = np.array(dephased_data)
    return dephased_state

st=result.states[50]
depahsed_st=random_dephasing(st)
print(st)

E_s_dephased= np.inner(np.conj(depahsed_st).T, np.matmul(H_list[2].full(),depahsed_st))
E_int_dephased= np.inner(np.conj(depahsed_st).T, np.matmul(H_list[3].full(),depahsed_st))
E_e_dephased= np.inner(np.conj(depahsed_st).T, np.matmul(H_list[4].full(),depahsed_st))

exp_val_time_dephased = [np.abs(E_s_dephased), np.abs(E_int_dephased), np.abs(E_e_dephased)]        
print(exp_val_time_dephased)


"""

def position(d1,H_list,state):

    H_int_s=H_list[6]
    eig_energ_int,eig_sta_int = H_int_s.eigenstates()


    state_temp=state
    #state_temp = qt.ptrace(result.states[frame], [0])

    weight = []
    for i in range(d1):
        weight.append(qt.expect(state_temp,eig_sta_int[i]))
    
    return weight,eig_energ_int



def copy_cat(d1,result,H_list,tlist,ind_1,ind_2,ind_3):
    #TODO clean    
    #1
    rho=result.states[ind_1]
    rho_s=qt.ptrace(rho,[0])
    eigenenergies,eigenstates=rho_s.eigenstates()
    l=len(eigenenergies)-1
    eig_1=eigenstates[l]
    eig_2=eigenstates[l-1]
    density_eig_1=eig_1*eig_1.dag()
    density_eig_2=eig_2*eig_2.dag()

    x1,eigenenergies=position(d1,H_list,density_eig_1)
    x2,eigenenergies=position(d1,H_list,density_eig_2)

    #2
    rho=result.states[ind_2]
    rho_s=qt.ptrace(rho,[0])
    eigenenergies,eigenstates=rho_s.eigenstates()
    l=len(eigenenergies)-1
    eig_1=eigenstates[l]
    eig_2=eigenstates[l-1]
    density_eig_1=eig_1*eig_1.dag()
    density_eig_2=eig_2*eig_2.dag()

    x3,eigenenergies=position(d1,H_list,density_eig_1)
    x4,eigenenergies=position(d1,H_list,density_eig_2)

    #3
    rho=result.states[ind_3]
    rho_s=qt.ptrace(rho,[0])
    eigenenergies,eigenstates=rho_s.eigenstates()
    l=len(eigenenergies)-1
    eig_1=eigenstates[l]
    eig_2=eigenstates[l-1]
    density_eig_1=eig_1*eig_1.dag()
    density_eig_2=eig_2*eig_2.dag()
    
    x5,eigenenergies=position(d1,H_list,density_eig_1)
    x6,eigenenergies=position(d1,H_list,density_eig_2)

    # Create a figure with three rows and two columns
    fig, axes = plt.subplots(3, 2, figsize=(6, 6))

    # Plot the first row
    axes[0, 0].plot(eigenenergies, x1,label=f't = {round(tlist[ind_1],2)}')
    axes[0, 0].set_title('First eigenstate')
    
    axes[0, 1].plot(eigenenergies, x2,label=f't = {round(tlist[ind_1],2)}')
    axes[0, 1].set_title('Second eigenstate')

    # Plot the second row
    axes[1, 0].plot(eigenenergies, x3,label=f't = {round(tlist[ind_2],2)}')
    axes[1, 1].plot(eigenenergies, x4,label=f't = {round(tlist[ind_2],2)}')

    # Plot the third row
    axes[2, 0].plot(eigenenergies, x5,label=f't = {round(tlist[ind_3],2)}')
    axes[2, 0].set_xlabel('Position')
    axes[2, 1].plot(eigenenergies, x6,label=f't = {round(tlist[ind_3],2)}')
    axes[2, 1].set_xlabel('Position')
    for i in range(3):
        for j in range(2):
            axes[i, j].legend(loc='upper right')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_einselection(result,ind_nb,d1):
    eigenenergies1_list=[]
    eigenenergies2_list=[]

    for i in range(ind_nb):
        rho_s=qt.ptrace(result.states[i], [0])
        eigenenergies, eigenstates = rho_s.eigenstates()
        eigenenergies1_list.append(eigenenergies[d1-1])
        eigenenergies2_list.append(eigenenergies[d1-2])

    plt.plot(eigenenergies1_list)
    plt.plot(eigenenergies2_list)
    plt.xlabel('Time step')
    plt.ylabel('Eigenenergy')
    plt.legend(['eigenvalue 1', 'eigenvalue 2'])
    plt.xscale('log')


def sum_real_off_diagonal_components(result,tlist):
    rho_s_list=[]
    for i in range(len(tlist)):
        rho_s=qt.ptrace(result.states[i], [0])
        rho_s_list.append(rho_s)

    sums = []
    for rho_s in rho_s_list:
        real_off_diagonal_sum = np.abs(np.sum(np.real(rho_s.full()) - np.real(np.diag(np.diag(rho_s.full())))))
        #print(real_off_diagonal_sum)
        sums.append(real_off_diagonal_sum)
    plt.plot(tlist,sums)
    plt.title('Reversible fluctuations in einselection')
    plt.xlabel('time')
    plt.ylabel('real off-diagonal terms')

    return sums