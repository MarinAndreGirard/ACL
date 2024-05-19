import numpy as np
import qutip as qt
import math
import matplotlib.pyplot as plt

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
    plt.xlabel("time index")
    plt.ylabel("VN entropy")



