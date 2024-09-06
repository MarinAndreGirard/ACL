import numpy as np
import qutip as qt
import math
import matplotlib.pyplot as plt

#TODO find a way to get ride of the runntime warning that comes everytime I run this

def compute_VN(result, time_index, subsystem_index=0):
    density_matrix = qt.ptrace(result.states[time_index], [subsystem_index])  # Calculate the density matrix at the specified time
    entropy = -np.sum(np.nan_to_num(np.log2(np.linalg.eigvals(density_matrix.full())) * np.linalg.eigvals(density_matrix.full())))
    return entropy

def compute_VN_time(result,tlist):
    von_neumann_entropy = []
    for time_index in range(len(tlist)):
        entropy = compute_VN(result, time_index, subsystem_index=0)
        von_neumann_entropy.append(entropy)
    return von_neumann_entropy

def plot_VN(result,tlist,log=0):
    v = compute_VN_time(result,tlist)
    plt.figure(figsize=(10, 2))
    plt.plot(tlist, v)
    if log == 1:
        plt.xscale('log')
    else:
        plt.xscale('linear')
    plt.title("VN entropy over time")
    plt.xlabel("time")
    plt.ylabel("VN entropy")


def compute_VN_rd(d1,d2,rho):

    # Step 1: Reshape the full density matrix into a 4D array for tracing
    rho_full_reshaped = rho.reshape([d1, d2, d1, d2])
    
    # Step 2: Trace out the second subsystem (d2)
    # This is equivalent to performing the partial trace over the second subsystem
    rho_subsystem = np.einsum('ijkl->ik', rho_full_reshaped)  # Partial trace over the second subsystem (d2)

    # Step 3: Convert the resulting reduced density matrix back into a Qobj
    rho_subsystem_qobj = qt.Qobj(rho_subsystem)

    # Step 4: Calculate the von Neumann entropy of the reduced density matrix
    entropy = qt.entropy_vn(rho_subsystem_qobj)
    return entropy

def plot_VN_numpy(d1,d2,result,tlist,log=0):
    # Store results
    density_matrices = []
    entropies = []

    for state in result.states:
        
        # Calculate the density matrix from the pure state
        rho = qt.ket2dm(state)  # This converts the pure state into a density matrix
        rho=rho.full()
        # Store the density matrix
        density_matrices.append(rho)

        # Calculate the von Neumann entropy
        entropy = compute_VN_rd(d1,d2,rho)  # Calculate von Neumann entropy
        entropies.append(entropy)

    # Plot entropy over time
    plt.plot(tlist, entropies, label="Subsystem Entropy")
    plt.xlabel('Time')
    plt.ylabel('Von Neumann Entropy')
    plt.title('Subsystem Entropy Over Time')
    if log == 1:
        plt.xscale('log')
    else:
        plt.xscale('linear')
    plt.grid(True)
    plt.legend()
    plt.show()
    return entropies
