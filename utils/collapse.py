import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from PIL import Image
#from overlap import compute_schmidt_full

def update(frames,eigenenergies_system_1,eigenstates_system_1,s_list):
    # Clear previous plot
    plt.clf()
    frames = frames+1
    
    s0=s_list[0][frame][0]

    schmidt_coefficients0 = [abs(np.vdot(s0, eigenstate)) ** 2 for eigenstate in eigenstates_system_1]

    plt.plot(eigenenergies_system_1, schmidt_coefficients0, marker='o', label=f'Energy {eigenenergies_system_1}')
    plt.title(f"Plot of the system state in its own energy eigenbasis")
    plt.xlabel("Eigenstates of the system")
    plt.text(0.95, 0.95, f"Frame: {frames}", horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)

def collapse_gif(eig, s_list,ind_nb):

    #Its taking the al time results of compute_schmidt states and filtering what it needs by itself. THis is decided 
    eigenenergies_system_1=eig[0]
    eigenstates_system_1=eig[3]

    # Create a figure
    fig = plt.figure(figsize=(10, 5))

    # Create the animation
    ani = FuncAnimation(fig, update,fargs=(eigenenergies_system_1,eigenstates_system_1,s_list), frames=ind_nb-1, interval=100)

    # Save the animation as a GIF
    ani.save(f'collapse.gif', writer='pillow')
    plt.close()

    return
