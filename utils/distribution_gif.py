import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from PIL import Image
#from overlap import compute_schmidt_full
from schmidt_solve import compute_schmidt_full


def update(frames,result,eigenstates_total,eigenenergies_total,info_list):#frames,result
    
    # Clear previous plot
    frames=frames+1
    EI=info_list[3]
    w=info_list[7]
    plt.clf()

    state = compute_schmidt_full(result,frames,1)
    state2 = compute_schmidt_full(result,frames,2)
    energy_coeff=[abs(np.vdot(state, eigenstate)) ** 2 for eigenstate in eigenstates_total]
    energy_coeff2=[abs(np.vdot(state2, eigenstate)) ** 2 for eigenstate in eigenstates_total]
    
    plt.plot(eigenenergies_total, energy_coeff)
    plt.plot(eigenenergies_total, energy_coeff2)
    plt.title(f"Plot of the probability that Schmidt1 and 2 are in the energy eigenstates for EI={EI} and w={w}")
    plt.xlabel("Eigenenergies of H_total")
    plt.ylabel("Probabilities")
    plt.ylim(0, 0.35)
    # Calculate the mean
    mean1 = np.sum(np.array(energy_coeff) * np.array(eigenenergies_total))
    mean2 = np.sum(np.array(energy_coeff2) * np.array(eigenenergies_total))
    st1_tst1 = np.mean((np.array(energy_coeff) * np.array(eigenenergies_total)-mean1)**2)
    st1_tst2 = np.mean((np.array(energy_coeff2) * np.array(eigenenergies_total)-mean2)**2)
    std1 = np.std(np.array(energy_coeff) * np.array(eigenenergies_total))
    std2 = np.std(np.array(energy_coeff2) * np.array(eigenenergies_total))
    # Add a vertical line at the mean for energy_coeff
    plt.axvline(x=mean1, color='b', linestyle='--')
    # Add a vertical line at the mean for energy_coeff2
    plt.axvline(x=mean2, color='r', linestyle='--')
    # Add a vertical line at the mean plus one standard deviation for energy_coeff
    plt.axvline(x=mean1 + st1_tst1, color='g', linestyle='--')
    # Add a vertical line at the mean minus one standard deviation for energy_coeff
    plt.axvline(x=mean1 - st1_tst1, color='g', linestyle='--')
    # Add a vertical line at the mean plus one standard deviation for energy_coeff2
    plt.axvline(x=mean2 + st1_tst2, color='c', linestyle='--')
    # Add a vertical line at the mean minus one standard deviation for energy_coeff2
    plt.axvline(x=mean2 - st1_tst2, color='c', linestyle='--')
    plt.legend(["Schmidt1", "Schmidt2", "Mean1", "Mean2", "Mean1 + Std1", "Mean1 - Std1", "Mean2 + Std2", "Mean2 - Std2"])
    # Add clock
    plt.text(0.95, 0.95, f"Frame: {frames}", horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)

def gif_distribution_eig_total(result,eig, info_list): #EI,w,result,eigenstates_total,eigenenergies_total,env,d1,d2,E_spacing,tmax,ind_nb
    
    #Get the necessary information
    eigenstates_total=eig[1]
    eigenenergies_total=eig[0]
    ind_nb=info_list[10]
    E_int=info_list[3]
    w=info_list[7]

    # Create a figure
    fig = plt.figure(figsize=(10, 5))

    # Create the animation
    ani = FuncAnimation(fig, update,fargs=(result,eigenstates_total,eigenenergies_total,info_list), frames=ind_nb-1, interval=100)

    # Save the animation as a GIF
    path = f'../outputs/gifs/distrib_param_{info_list}.gif'
    ani.save(path, writer='pillow')
    plt.close()

    return path
