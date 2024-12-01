import sys
import os

# modify syspath directory to sys.path

package_path = os.path.abspath(os.path.join('..','core'))
if package_path not in sys.path:
    sys.path.append(package_path)

package_path = os.path.abspath(os.path.join('..','utils'))
if package_path not in sys.path:
    sys.path.append(package_path)

package_path = os.path.abspath(os.path.join('..'))
if package_path not in sys.path:
    sys.path.append(package_path)

import numpy as np
import qutip as qt
import math
import matplotlib.pyplot as plt
import pickle

from IPython.display import HTML # both needed to disply gifs
from PIL import Image

# Import modules from the package
from core import create_hamiltonian as ch
from core import create_state as cs
from core import time_evo
from core import load_param
from core import load_result
from core import load_tlist
from core.eigen_ener_states import eigen_ener_states as eig


plt.rc('text', usetex=False)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = 16             # Title font size
plt.rcParams['axes.titleweight'] = 'bold'       # Title font weight
plt.rcParams['axes.labelsize'] = 14             # X and Y label font size
plt.rcParams['axes.labelweight'] = 'bold'       # X and Y label font weight
#plt.rcParams['axes.grid'] = True                # Enable grid
#plt.rcParams['grid.alpha'] = 0.7                # Grid transparency
#plt.rcParams['grid.linestyle'] = '--'           # Grid line style
#plt.rcParams['grid.color'] = 'gray'             # Grid color

# Tick settings
plt.rcParams['xtick.labelsize'] = 12            # X tick label size
plt.rcParams['ytick.labelsize'] = 12            # Y tick label size
plt.rcParams['xtick.direction'] = 'in'          # X tick direction
plt.rcParams['ytick.direction'] = 'in'          # Y tick direction
plt.rcParams['xtick.major.size'] = 6            # X major tick size
plt.rcParams['ytick.major.size'] = 6            # Y major tick size

# Legend settings
plt.rcParams['legend.fontsize'] = 12            # Legend font size
plt.rcParams['legend.frameon'] = True           # Enable legend frame
plt.rcParams['legend.framealpha'] = 0.9         # Legend frame transparency
plt.rcParams['legend.loc'] = 'best'             # Legend location

# Line and marker settings
plt.rcParams['lines.linewidth'] = 2             # Line width
plt.rcParams['lines.markersize'] = 6            # Marker size

custom_colors = ['#1c4587', '#e6194B', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)


ar=np.array
kr=np.kron
idn=np.identity
sx=ar([[0,1],[1,0]])
sy=ar([[0,-1j],[1j,0]])
sz=ar([[1,0],[0,-1]])
id2=ar([[1,0],[0,1]])

def random_hermitian_matrix(size):
    real_part = np.random.randn(size, size)
    imag_part = np.random.randn(size, size) * 1j
    A = real_part + imag_part
    return (A + A.conj().T) / 2  # Make it Hermitian

def create_H_not_2_local(n):
    size=2**n
    hermitian_matrix = random_hermitian_matrix(size)
    
    # Ensure the diagonal is real
    #np.fill_diagonal(hermitian_matrix, np.random.rand(size))
    
    eigenvalues = np.linalg.eigvals(hermitian_matrix)
    max_eigenvalue = np.max(np.abs(eigenvalues))  # Largest eigenvalue by magnitude
    return hermitian_matrix / max_eigenvalue


def create_rando_n_matrix(n):
    #create a compelx random matrix of size n
    rd = np.random.rand(n, n)
    return rd

def create_2_local(n):
    r1=create_rando_n_matrix(n)
    r2=create_rando_n_matrix(n)
    r3=create_rando_n_matrix(n)
    r4=create_rando_n_matrix(n)
    r5=create_rando_n_matrix(n)
    r6=create_rando_n_matrix(n)
    r7=create_rando_n_matrix(n)
    #we then interpret the off diagonal terms of these random matrices as the coupling terms
    #Using r1 we define the zz terms
    H=np.zeros((2**(n), 2**(n)), dtype=np.complex64)
    paulis_list=[sx,sy,sz]
    r_list=[r1,r2,r3]
    for i in range(n):
        for j in range(n):
            if j>i:
                for pauli, r in zip(paulis_list, r_list):
                    s1=kr(kr(idn(2**i),pauli),idn(2**(n-i-1)))
                    s2=kr(kr(idn(2**j),pauli),idn(2**(n-j-1)))
                    H+=r[i,j]*s1@s2
                s1=kr(kr(idn(2**i),sz),idn(2**(n-i-1)))
                s2=kr(kr(idn(2**j),sx),idn(2**(n-j-1)))
                H+=r4[i,j]*s1@s2
                s1=kr(kr(idn(2**i),sx),idn(2**(n-i-1)))
                s2=kr(kr(idn(2**j),sz),idn(2**(n-j-1)))
                H+=r5[i,j]*s1@s2
                s1=kr(kr(idn(2**i),sz),idn(2**(n-i-1)))
                s2=kr(kr(idn(2**j),sy),idn(2**(n-j-1)))
                H+=r6[i,j]*s1@s2
                s1=kr(kr(idn(2**i),sy),idn(2**(n-i-1)))
                s2=kr(kr(idn(2**j),sz),idn(2**(n-j-1)))
                H+=r7[i,j]*s1@s2

    eigenvalues = np.linalg.eigvals(H)
    max_eigenvalue = np.max(np.abs(eigenvalues))  # Largest eigenvalue by magnitude
    return H / max_eigenvalue

def create_1_local(n):
    #r1 an array of random complex numbers of size n
    r1 = np.random.rand(n)
    r2 = np.random.rand(n)
    r3 = np.random.rand(n)

    H=np.zeros((2**(n), 2**(n)), dtype=np.complex64)
    paulis_list=[sx,sy,sz]
    r_list=[r1,r2,r3]
    for i in range(n):
        for pauli, r in zip(paulis_list, r_list):
            s1=kr(kr(idn(2**i),pauli),idn(2**(n-i-1)))
            H+=r[i]*s1
    eigenvalues = np.linalg.eigvals(H)
    max_eigenvalue = np.max(np.abs(eigenvalues))  # Largest eigenvalue by magnitude
    return H / max_eigenvalue

def create_H_2_local(n,a1=0.5,a2=0.75,a3=0.2):
    I_e=qt.qeye(2**(n-1))
    I_s=qt.qeye(2)
    
    H_s = qt.Qobj(sz)
    H_s = a1*qt.tensor(H_s,I_e)

    #H_e=create_He_i_2_local(n-1)
    H_e=create_2_local(n-1) #For the environment, we use a 2-local Hamiltonian. thats the whole point.
    H_e=qt.Qobj(H_e)
    H_e = a2*qt.tensor(I_s,H_e)
    
    H_ei=create_1_local(n-1) # For the interaction, the environment term is made of 1-local terms since we want HI to be 2-local.
    H_ei=qt.Qobj(H_ei)
    H_I=a3*qt.tensor(qt.Qobj(sz),H_ei)
    
    H = H_s+H_e+H_I

    eigenvalues = H.eigenenergies()
    max_eigenvalue = max(eigenvalues)
    H=H/max_eigenvalue
    
    return H



def create_H_non_local(n,a1=0.5,a2=0.75,a3=0.2):
    I_e=qt.qeye(2**(n-1))
    I_s=qt.qeye(2)
    
    H_s = qt.Qobj(sz)
    H_s = a1*qt.tensor(H_s,I_e)

    H_e=create_H_not_2_local(n-1)
    H_e=qt.Qobj(H_e)
    H_e = a2*qt.tensor(I_s,H_e)
    
    H_ei=create_H_not_2_local(n-1)
    H_ei=qt.Qobj(H_ei)
    H_I=a3*qt.tensor(qt.Qobj(sz),H_ei)
    
    H = H_s+H_e+H_I
    
    eigenvalues = H.eigenenergies()
    max_eigenvalue = max(eigenvalues)
    H=H/max_eigenvalue
    
    return H

def create_state_2_local(n_e):
    w=0.3
    # Create the superposition state for the system
    system_superposition_state = (np.sqrt(w)*qt.basis(2, 0) + np.sqrt(1-w)*qt.basis(2, 1)).unit()
    random_state = qt.rand_ket(2**n_e)
    state = qt.tensor(system_superposition_state, random_state)
    return state


def time_evo(H,state,log=0,tmax=10,ind_nb=100,file_name="default"):

    tlist = np.linspace(0, tmax, ind_nb) # Linear spacing
    if log == 0:
        tlist = np.linspace(0, tmax, ind_nb)  # Linear spacing
    elif log == 1:
        tlist = np.logspace(np.log10(1), np.log10(tmax+1), ind_nb)-1  # Logarithmic spacing
    else:
        raise ValueError("Invalid value for 'log'. It should be either 0 or 1.")
    info_list=[tmax, ind_nb,log,tlist]
    
    # Perform time evolution of the combined system
    result = qt.mesolve(H, state, tlist, [], [])

    # Save outputs in a .txt file
    outputs_dir = "outputs/simulation_results"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    # Save parameters in a .txt file
    params_file_path = os.path.join(outputs_dir, "params_" + file_name)
    with open(params_file_path, "w") as f:
        tmax, ind_nb,log
        f.write(f"tmax === {tmax}\n")
        f.write(f"ind_nb === {ind_nb}\n")
        f.write(f"log === {log}\n")
    
    # Save parameters in a .txt file
    tlist_file_path = os.path.join(outputs_dir, "tlist_" + file_name)
    np.save(tlist_file_path, tlist)
    
    # Save result in a .txt file
    result_file_path = os.path.join(outputs_dir, "result_" + file_name)
    qt.qsave(result, result_file_path)
    
    # Save H_list in a .txt file
    
    H_path = os.path.join(outputs_dir, "H_list_" + file_name)
    qt.qsave(H, H_path)
    
    return result, tlist, H, state, info_list

def compute_schmidt_states_new(result, time_index,d1,d2):
    global_state = result.states[time_index]
    density_matrix = qt.ptrace(global_state, [0]) # Calculate the density matrix at the specified time
    eigenvalues, eigenstates = density_matrix.eigenstates() # Compute the eigenstates and eigenvalues of the density matrix
    eigenstates = [np.array(state.full()) for state in eigenstates]
    schmidt_states_s = []
    schmidt_states_e = []
    schmidt_values = []
    i=0
    for state_, eigenvalue in zip(eigenstates, eigenvalues):
        schmidt_values.append(eigenvalue)
        if eigenvalue < 10e-14:
            # If the eigenvalue is zero, set the Schmidt state to a zero vector
            schmidt_states_s.append(np.zeros_like(state_))
        else:
            #print(f"state {state}")
            i=i+1
            #N=qt.norm(state_)
            N=abs(np.vdot(state_,state_))
            schmidt_states_s.append(state_/np.sqrt(N)) # Normalize

    # Sort the Schmidt states by eigenvalue in descending order
    schmidt_states_s, schmidt_values = zip(*sorted(zip(schmidt_states_s, schmidt_values), key=lambda x: -x[1]))
    #d=np.size(global_state)
    #print(f"d {d}")
    #d1 = np.size(schmidt_states_s[0])
    #d2=d//d1
    #compute the schmidt states of the environement.
    schmidt_states_e = []
    I = np.eye(d2)
    #to get schmidt_env, we use that schmidt 1 and 2 of the system are |phi1> and |phi2>. 
    #Then we have that the global state can always be written as |psi>=s1|phi1>|a1>+s2|phi2>|a2>  
    for j in range(i):
        #step 1: get |phi_i>|ai>
        state = schmidt_states_s[j] #getting |phi_i>, is normalized
        P_a_state = np.kron(np.outer(state,state.conjugate().T),I) #def projector |phi_i><phi_i|xId, np.outer transposes the second one
       
        temp = np.dot(P_a_state,global_state) #We apply the projector on the global state: P|psi>=s1|phi_i>|a_i> and normalize. vdot is conjugate on first one. To normalize /schmidt_values[j] potentially
        temp = temp.flatten()
        #temp1=temp.full()
        #print(f"temp1 {temp1}")
        #step 2:We have |phi_i>|a_i>. We know |phi_i> and want |a_i>. |phi_i>|a_i> = (phi^1_i|a_i>,phi^2_i|a_i>, ...)
        #find the first nonzero coeff of |phi_i> and use it to extract |a_i>
        nonzero_index = np.nonzero(state)[0][0] #will need to know the index of a nonzero value in |phi_i>
        #now extract a d2 sized vector from thatd2*d1 sized vector
        temp2 = temp[nonzero_index*d2:(nonzero_index+1)*d2] #for k the nnzero index, this is phi^k_i|a_i>
        temp3=temp2/state[nonzero_index]
        N = abs(np.vdot(temp3[0].full(),temp3[0].full()))
        
        schmidt_states_e.append(temp3/np.sqrt(N))

    return schmidt_states_s,schmidt_states_e,schmidt_values

def compute_schmidt_full(result,idx,s=1):
    ss, se, sv = compute_schmidt_states_new(result, idx,d1,d2)
    if s==1:
        a = ss[0] #schmidt 1 on system 1
        a = np.squeeze(a)
        b = se[0] #schmidt 1 on system 2
        b=np.squeeze(b)
        g = np.outer(a,b).flatten()
        g=np.squeeze(g)
    elif s==2:
        a = ss[1] #schmidt 2 on system 1
        a = np.squeeze(a)
        b = se[1] #schmidt 2 on system 2
        b=np.squeeze(b)
        g = np.outer(a,b).flatten()
        g=np.squeeze(g)
    return g

def compute_schmidt_states_all_time(result, ind_nb,d1,d2):

    #TODO
    #-Another thing i want to do here is to outpute for all time the compute_schmidt_full() without redundancy in the computation
    #-Make it do both env and syst.
    schmidt_states_s_tlist=[]
    schmidt_states_e_tlist=[]
    schmidt_values_tlist=[]
    schmidt_full_tlist=[]
    

    for time_index in range(ind_nb):
        ss, se, sv = compute_schmidt_states_new(result, time_index,d1,d2)
        
        schmidt_states_s_tlist.append(np.squeeze(ss))
        schmidt_states_e_tlist.append(se)
        schmidt_values_tlist.append(sv)

        #making the full schmdits #TODO this will have to be modified to make it work for more than 2 superpositions
        a1 = ss[0].flatten() #schmidt 1 on system 1
        a1 = np.squeeze(a1)
        b1 = se[0].flatten() #schmidt 1 on system 2
        b1 = np.squeeze(b1)
        g1 = np.outer(a1,b1).flatten()
        g1 = np.squeeze(g1)
        a2 = ss[1].flatten() #schmidt 2 on system 1
        a2 = np.squeeze(a2)
        if time_index != 0:
            b2 = se[1].flatten() #schmidt 2 on system 2
            b2=np.squeeze(b2)
        else:
            b2 = np.zeros_like(b1)
        g2 = np.outer(a2,b2).flatten()
        g2=np.squeeze(g2)
        g=[g1,g2]
        schmidt_full_tlist.append(g)

    return schmidt_states_s_tlist,schmidt_states_e_tlist,schmidt_values_tlist,schmidt_full_tlist

def get_VN_entropy(d1,d2,result,tlist,log=0):
    # Store results
    density_matrices = []
    entropies = []
    for state in result.states:
        
        # Calculate the density matrix from the pure state
        rho = qt.ket2dm(state)  # This converts the pure state into a density matrix
        density_matrix_qobj = qt.Qobj(rho, dims=[[d1, d2], [d1, d2]])
        traced_system = qt.ptrace(density_matrix_qobj, 1)  # Keep qubits 2 and 3
        entropy=qt.entropy_vn(traced_system)
        entropies.append(entropy)
    # Plot entropy over time
    return entropies




def run_simulation(create_H_func, file_suffix, file_name="default", n_e=8, log=0, tmax=40, ind_nb=200,a1=0.5,a2=0.75,a3=0.2):
    # Construct the filename
    file_name_2 = f"{file_name}_{file_suffix}"
    
    # Create Hamiltonian and initial state
    H = create_H_func(n_e+1,a1,a2,a3)
    state = create_state_2_local(n_e)
    # Run time evolution and load results
    result, tlist, H, state, info_list = time_evo(H, state, log, tmax, ind_nb, file_name_2)
    tmax, ind_nb, log = load_param(file_name_2)
    result = load_result(file_name_2)
    tlist = load_tlist(file_name_2)
    # Compute Schmidt states and entropies
    #s_list = compute_schmidt_states_all_time(result, ind_nb, 2, 2**n_e)
    entropies = get_VN_entropy(2, 2**n_e, result, tlist, log=0)

    #get the eigenvalues of rho_s over time
    eigenvalues_list = []
    for i in range(0,ind_nb):
        density_matrix = qt.ptrace(result.states[i], [0]) # Calculate the density matrix at the specified time
        eigenvalues, eigenstates = density_matrix.eigenstates() # Compute the eigenstates and eigenvalues of the density matrix
        eigenvalues_list.append(eigenvalues)

    return tlist, entropies, eigenvalues_list


#Run simulation shouldn't be calculating the entropies and eigenvalues. this should be done in a different function that calls upon the outputs we saved.
#I now want to test if I can call upon the outputs we saved to calculate the entropies and eigenvalues.

def get_entrop_and_eigen(file_suffix,file_name,n_e):
    file_name_2=f"{file_name}_{file_suffix}"
    tmax, ind_nb, log = load_param(file_name_2)
    result = load_result(file_name_2)
    tlist = load_tlist(file_name_2)
    
    # Compute Schmidt states and entropies
    #s_list = compute_schmidt_states_all_time(result, ind_nb, 2, 2**n_e)
    entropies = get_VN_entropy(2, 2**n_e, result, tlist, log=0)

    #get the eigenvalues of rho_s over time
    eigenvalues_list = []
    for i in range(0,ind_nb):
        density_matrix = qt.ptrace(result.states[i], [0]) # Calculate the density matrix at the specified time
        eigenvalues, eigenstates = density_matrix.eigenstates() # Compute the eigenstates and eigenvalues of the density matrix
        eigenvalues_list.append(eigenvalues)

    return tlist, entropies, eigenvalues_list


def plot_function(file_name_local,file_name_non_local,file_suffixes=[1,2],n_e=8):
    file_name = file_name_local
    # Local Hamiltonian entropies
    local_entropies = []
    local_eigenvalues = []
    for suffix in file_suffixes:
        tlist, entropies, eigenvalues_list = get_entrop_and_eigen(suffix,file_name,n_e)
        local_entropies.append(entropies)
        local_eigenvalues.append(eigenvalues_list)

    file_name = file_name_non_local
    # Non-local Hamiltonian entropies
    non_local_entropies = []
    non_local_eigenvalues = []
    for suffix in file_suffixes:
        tlist, entropies, eigenvalues_list= get_entrop_and_eigen(suffix,file_name,n_e)
        non_local_entropies.append(entropies)
        non_local_eigenvalues.append(eigenvalues_list)

    # Plot the results
    from matplotlib import cm

    blues = cm.Blues(np.linspace(0.4, 0.8, len(file_suffixes)))  # Blue shades for the first 4 (local)
    reds = cm.Reds(np.linspace(0.4, 0.8, len(file_suffixes)))    # Red shades for the second 4 (non-local)

    l=len(file_suffixes)

    # Plot for local Hamiltonian
    for i in range(l-1):
        plt.plot(tlist, local_entropies[i], color=blues[i])
    plt.plot(tlist, local_entropies[l-1], color=blues[l-1], label="Local Entropy")

    # Plot for non-local Hamiltonian
    for i in range(l-1):
        plt.plot(tlist, non_local_entropies[i], color=reds[i])
    plt.plot(tlist, non_local_entropies[l-1], color=reds[l-1], label="Non-local Entropy")

    plt.xlabel(r'$t$')
    plt.ylabel(r'$S_{\rho_s}$')
    #plt.title('Von Neumann Entropy over time')
    plt.grid(True)
    plt.legend()
    name=file_name_local+"VN"+".pickle"
    with open(name, 'wb') as f:
        pickle.dump(plt.gcf(), f)  # Save the current figure (gcf)
    plt.show()

    plt.plot(0, local_eigenvalues[l-1][0][0], color=blues[l-1], label="Eigenvalues local case")
    plt.plot(0, non_local_eigenvalues[l-1][0][0], color=reds[l-1], label="Eigenvalues non-local case")

    for i in range(l):
        plt.plot(tlist, local_eigenvalues[i], color=blues[i])
    

    # Plot for non-local Hamiltonian
    for i in range(l):
        plt.plot(tlist, non_local_eigenvalues[i], color=reds[i])
    
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\lambda$')
    #plt.title('Eigenvalues over time')
    plt.grid(True)
    plt.legend()
    name=file_name_local+"eigval"+".pickle"
    with open(name, 'wb') as f:
        pickle.dump(plt.gcf(), f)  # Save the current figure (gcf)
    plt.show()

def plot_function_simple(file_name_local,file_suffixes=[1,2],n_e=8):
    file_name = file_name_local
    # Local Hamiltonian entropies
    local_entropies = []
    local_eigenvalues = []
    for suffix in file_suffixes:
        tlist, entropies, eigenvalues_list = get_entrop_and_eigen(suffix,file_name,n_e)
        local_entropies.append(entropies)
        local_eigenvalues.append(eigenvalues_list)

    # Plot the results
    from matplotlib import cm

    blues = cm.Blues(np.linspace(0.4, 0.8, len(file_suffixes)))  # Blue shades for the first 4 (local)

    l=len(file_suffixes)

    # Plot for local Hamiltonian
    for i in range(l-1):
        plt.plot(tlist, local_entropies[i], color=blues[i])
    plt.plot(tlist, local_entropies[l-1], color=blues[l-1], label="Local Entropy")

    plt.xlabel(r'$t$')
    plt.ylabel(r'$S_{\rho_s}$')
    #plt.title('Von Neumann Entropy over time')
    plt.grid(True)
    #plt.legend()
    name=file_name+"VN"+".pickle"
    with open(name, 'wb') as f:
        pickle.dump(plt.gcf(), f)  # Save the current figure (gcf)

    plt.show()

    plt.plot(0, local_eigenvalues[l-1][0][0], color=blues[l-1], label="Eigenvalues local case")

    for i in range(l):
        plt.plot(tlist, local_eigenvalues[i], color=blues[i])
        
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\lambda$')
    #plt.title('Eigenvalues over time')
    plt.grid(True)
    #plt.legend()
    name=file_name+"eigval"+".pickle"
    with open(name, 'wb') as f:
        pickle.dump(plt.gcf(), f)  # Save the current figure (gcf)
    plt.show()



ar=np.array
kr=np.kron
idn=np.identity
sx=ar([[0,1],[1,0]])
sy=ar([[0,-1j],[1j,0]])
sz=ar([[1,0],[0,-1]])
id2=ar([[1,0],[0,1]])

def random_hermitian_matrix(size):
    real_part = np.random.randn(size, size)
    imag_part = np.random.randn(size, size) * 1j
    A = real_part + imag_part
    return (A + A.conj().T) / 2  # Make it Hermitian

def create_H_not_2_local(n):
    size=2**n
    hermitian_matrix = random_hermitian_matrix(size)
    
    # Ensure the diagonal is real
    #np.fill_diagonal(hermitian_matrix, np.random.rand(size))
    
    eigenvalues = np.linalg.eigvals(hermitian_matrix)
    max_eigenvalue = np.max(np.abs(eigenvalues))  # Largest eigenvalue by magnitude
    return hermitian_matrix / max_eigenvalue


def create_rando_n_matrix(n):
    #create a compelx random matrix of size n
    rd = np.random.rand(n, n)
    return rd

def create_2_local(n):
    r1=create_rando_n_matrix(n)
    r2=create_rando_n_matrix(n)
    r3=create_rando_n_matrix(n)
    r4=create_rando_n_matrix(n)
    r5=create_rando_n_matrix(n)
    r6=create_rando_n_matrix(n)
    r7=create_rando_n_matrix(n)
    #we then interpret the off diagonal terms of these random matrices as the coupling terms
    #Using r1 we define the zz terms
    H=np.zeros((2**(n), 2**(n)), dtype=np.complex64)
    paulis_list=[sx,sy,sz]
    r_list=[r1,r2,r3]
    for i in range(n):
        for j in range(n):
            if j>i:
                for pauli, r in zip(paulis_list, r_list):
                    s1=kr(kr(idn(2**i),pauli),idn(2**(n-i-1)))
                    s2=kr(kr(idn(2**j),pauli),idn(2**(n-j-1)))
                    H+=r[i,j]*s1@s2
                s1=kr(kr(idn(2**i),sz),idn(2**(n-i-1)))
                s2=kr(kr(idn(2**j),sx),idn(2**(n-j-1)))
                H+=r4[i,j]*s1@s2
                s1=kr(kr(idn(2**i),sx),idn(2**(n-i-1)))
                s2=kr(kr(idn(2**j),sz),idn(2**(n-j-1)))
                H+=r5[i,j]*s1@s2
                s1=kr(kr(idn(2**i),sz),idn(2**(n-i-1)))
                s2=kr(kr(idn(2**j),sy),idn(2**(n-j-1)))
                H+=r6[i,j]*s1@s2
                s1=kr(kr(idn(2**i),sy),idn(2**(n-i-1)))
                s2=kr(kr(idn(2**j),sz),idn(2**(n-j-1)))
                H+=r7[i,j]*s1@s2

    eigenvalues = np.linalg.eigvals(H)
    max_eigenvalue = np.max(np.abs(eigenvalues))  # Largest eigenvalue by magnitude
    return H / max_eigenvalue

def create_1_local(n):
    #r1 an array of random complex numbers of size n
    r1 = np.random.rand(n)
    r2 = np.random.rand(n)
    r3 = np.random.rand(n)

    H=np.zeros((2**(n), 2**(n)), dtype=np.complex64)
    paulis_list=[sx,sy,sz]
    r_list=[r1,r2,r3]
    for i in range(n):
        for pauli, r in zip(paulis_list, r_list):
            s1=kr(kr(idn(2**i),pauli),idn(2**(n-i-1)))
            H+=r[i]*s1
    eigenvalues = np.linalg.eigvals(H)
    max_eigenvalue = np.max(np.abs(eigenvalues))  # Largest eigenvalue by magnitude
    return H / max_eigenvalue

def create_H_2_local(n,a1=0.5,a2=0.75,a3=0.2,system_mode=0,interaction_mode=0):
    I_e=qt.qeye(2**(n-1))
    I_s=qt.qeye(2)
    if system_mode==0:
        H_s = qt.Qobj(sz)
    else: 
        H_s = qt.Qobj(sy)
    
    H_s = a1*qt.tensor(H_s,I_e)

    #H_e=create_He_i_2_local(n-1)
    H_e=create_2_local(n-1) #For the environment, we use a 2-local Hamiltonian. thats the whole point.
    H_e=qt.Qobj(H_e)
    H_e = a2*qt.tensor(I_s,H_e)
    
    H_ei=create_1_local(n-1) # For the interaction, the environment term is made of 1-local terms since we want HI to be 2-local.
    H_ei=qt.Qobj(H_ei)
    
    if interaction_mode==0:
        H_I=a3*qt.tensor(qt.Qobj(sz),H_ei)
    else:
        H_I=a3*qt.tensor(qt.Qobj(id),H_ei)
    
    H = H_s+H_e+H_I

    eigenvalues = H.eigenenergies()
    max_eigenvalue = max(eigenvalues)
    H=H/max_eigenvalue
    
    return H



def create_H_non_local(n,a1=0.5,a2=0.75,a3=0.2,system_mode=0,interaction_mode=0):
    I_e=qt.qeye(2**(n-1))
    I_s=qt.qeye(2)

    if system_mode==0:
        H_s = qt.Qobj(sz)
    else: 
        H_s = qt.Qobj(sy)
    
    H_s = a1*qt.tensor(H_s,I_e)

    H_e=create_H_not_2_local(n-1)
    H_e=qt.Qobj(H_e)
    H_e = a2*qt.tensor(I_s,H_e)
    

    H_ei=create_H_not_2_local(n-1)
    H_ei=qt.Qobj(H_ei)
    if interaction_mode==0:
        H_I=a3*qt.tensor(qt.Qobj(sz),H_ei)
    else:
        H_I=a3*qt.tensor(qt.Qobj(id),H_ei)
    
    H = H_s+H_e+H_I
    
    eigenvalues = H.eigenenergies()
    max_eigenvalue = max(eigenvalues)
    H=H/max_eigenvalue
    
    return H


def create_state_2_local(n_e):
    w=0.3
    # Create the superposition state for the system
    system_superposition_state = (np.sqrt(w)*qt.basis(2, 0) + np.sqrt(1-w)*qt.basis(2, 1)).unit()
    random_state = qt.rand_ket(2**n_e)
    state = qt.tensor(system_superposition_state, random_state)
    return state

def create_e_state(n_e):
    random_state = qt.rand_ket(2**n_e)
    return random_state

def create_H_non_local_H0(n,a1=0.5,a2=0.75,a3=0.2,system_mode=0,interaction_mode=0):
    I_e=a1*qt.qeye(2**(n-1))

    H_e=create_H_not_2_local(n-1)
    H_e=a3*qt.Qobj(H_e)
    
    H_ei=create_H_not_2_local(n-1)
    H_ei=a3*qt.Qobj(H_ei)
    
    H = I_e+H_ei+H_e
    
    eigenvalues = H.eigenenergies()
    max_eigenvalue = max(eigenvalues)
    H=H/max_eigenvalue
    
    return H
def create_H_non_local_H1(n,a1=0.5,a2=0.75,a3=0.2,system_mode=0,interaction_mode=0):
    I_e=a1*qt.qeye(2**(n-1))

    H_e=create_H_not_2_local(n-1)
    H_e=a3*qt.Qobj(H_e)
    
    H_ei=create_H_not_2_local(n-1)
    H_ei=a3*qt.Qobj(H_ei)
    
    H = -I_e-H_ei+H_e
    
    eigenvalues = H.eigenenergies()
    max_eigenvalue = max(eigenvalues)
    H=H/max_eigenvalue
    
    return H

def create_H_2_local_H0(n,a1=0.5,a2=0.75,a3=0.2,system_mode=0,interaction_mode=0):
    I_e=qt.qeye(2**(n-1))    

    #H_e=create_He_i_2_local(n-1)
    H_e=a1*create_2_local(n-1) #For the environment, we use a 2-local Hamiltonian. thats the whole point.
    H_e=a2*qt.Qobj(H_e)
    
    H_ei=create_1_local(n-1) # For the interaction, the environment term is made of 1-local terms since we want HI to be 2-local.
    H_ei=a3*qt.Qobj(H_ei)
    
    H = I_e+H_ei+H_e

    eigenvalues = H.eigenenergies()
    max_eigenvalue = max(eigenvalues)
    H=H/max_eigenvalue
    
    return H

def create_H_2_local_H1(n,a1=0.5,a2=0.75,a3=0.2,system_mode=0,interaction_mode=0):
    I_e=qt.qeye(2**(n-1))    

    #H_e=create_He_i_2_local(n-1)
    H_e=a1*create_2_local(n-1) #For the environment, we use a 2-local Hamiltonian. thats the whole point.
    H_e=a2*qt.Qobj(H_e)
    
    H_ei=create_1_local(n-1) # For the interaction, the environment term is made of 1-local terms since we want HI to be 2-local.
    H_ei=a3*qt.Qobj(H_ei)
    
    H = -I_e-H_ei+H_e
    
    eigenvalues = H.eigenenergies()
    max_eigenvalue = max(eigenvalues)
    H=H/max_eigenvalue
    
    return H

# Function to calculate and plot eigenvalue spectrum for H^n
def plot_spectrum_powers(H1,H2, max_power=5):
    fig, axes = plt.subplots(1, max_power, figsize=(15, 4))

    for n in range(1, max_power + 1):
        # Compute H^n
        H_n = (H1@H2)**n

        # Get the eigenvalues
        eigenenergies, _ = H_n.eigenstates()

        # Plot the spectrum
        ax = axes[n-1]
        ax.hist(eigenenergies, bins=50, alpha=0.7, color='b')
        ax.set_title(f"Spectrum of H^{n}")
        ax.set_xlabel("Energy")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()



