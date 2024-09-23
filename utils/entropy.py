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


def plot_VN_numpy(d1,d2,result,tlist,log=0):
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