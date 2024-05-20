import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from PIL import Image
#from overlap import compute_schmidt_full
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def update(frame, eigenenergies_system_1, eigenstates_system_1, s_list, min_schmidt, max_schmidt):
    # Clear previous plot
    plt.clf()
    
    s0 = s_list[0][frame][0]
    schmidt_coefficients0 = [abs(np.vdot(s0, eigenstate)) ** 2 for eigenstate in eigenstates_system_1]

    plt.plot(eigenenergies_system_1, schmidt_coefficients0, marker='o', label=f'Energy {eigenenergies_system_1}')
    plt.title(f"Plot of the system state in its own energy eigenbasis")
    plt.xlabel("Eigenstates of the system")
    plt.ylabel("Schmidt Coefficients")
    plt.ylim(min_schmidt, max_schmidt)  # Set fixed y-axis limits

    plt.text(0.95, 0.95, f"Frame: {frame}", horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)

def collapse_gif(eig, s_list, ind_nb):
    eigenenergies_system_1 = eig[0]
    eigenstates_system_1 = eig[3]

    # Compute global min and max values for Schmidt coefficients to fix y-axis limits
    all_schmidt_coefficients = []
    for frame in range(ind_nb):
        s0 = s_list[0][frame][0]
        schmidt_coefficients0 = [abs(np.vdot(s0, eigenstate)) ** 2 for eigenstate in eigenstates_system_1]
        all_schmidt_coefficients.extend(schmidt_coefficients0)

    min_schmidt = min(all_schmidt_coefficients)
    max_schmidt = max(all_schmidt_coefficients)

    # Create a figure
    fig = plt.figure(figsize=(10, 5))

    # Create the animation
    ani = FuncAnimation(fig, update, fargs=(eigenenergies_system_1, eigenstates_system_1, s_list, min_schmidt, max_schmidt), frames=ind_nb, interval=100)

    # Save the animation as a GIF
    path = f'../outputs/gifs/collapse.gif'
    ani.save(path, writer='pillow')
    plt.close()

    return path
