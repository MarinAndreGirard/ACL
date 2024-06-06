import numpy as np
import qutip as qt
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from PIL import Image


#Defining the overlap of states in probability space
# a measure of similarity of weights in the energie eigenbasis

def get_p_s2(state,eigenstates_total):
    p=[abs(np.vdot(state, eigenstate)) ** 2  for eigenstate in eigenstates_total]
    return p


def p_overlap(state1,state2,eigenstates_total):
    sqrt_p1 = np.sqrt(get_p_s2(state1,eigenstates_total))
    sqrt_p2 = np.sqrt(get_p_s2(state2,eigenstates_total))
    overlap = np.dot(sqrt_p1, sqrt_p2)
    return overlap

def overlap(tlist,result,H_list,s_list,eig):

    H_total=H_list[1]
    o01 = []
    o02 = []
    o12 = []
    eigenstates_total = eig[1]
    #eigenenergies_total, eigenstates_total = H_total.eigenstates()
    for idx in range(len(tlist)-2):
        #s1=compute_schmidt_full(result,idx+1,1)
        #s2=compute_schmidt_full(result,idx+1,2)
        s1=s_list[3][idx][0]
        s2=s_list[3][idx][1]
        global_state = result.states[idx+1]
        #s3=compute_schmidt_full(result,idx,3)
        o01.append(p_overlap(global_state,s1,eigenstates_total))
        o02.append(p_overlap(global_state,s2,eigenstates_total))
        o12.append(p_overlap(s1,s2,eigenstates_total))

    return o01, o02, o12


def update(frames,eigenstates_total,eigenenergies_total,s_full_list,info_list,zoom):
    # Clear previous plot
    frames=frames+1
    EI=info_list[3]
    w=info_list[7]
    plt.clf()
 
    #plot of the values of the overlap of first term in the first schmidt wit the first term in the second schmitd  in the total energy eigenbasis for all states.

    s1 = s_full_list[frames][0]
    s2 = s_full_list[frames][1]

    o1=[abs(np.vdot(s1, eigenstate)) ** 2  for eigenstate in eigenstates_total]
    o2=[abs(np.vdot(s2, eigenstate)) ** 2  for eigenstate in eigenstates_total]
    o=[o1[i]*o2[i] for i in range(len(o1))]
    if zoom == True:
        plt.plot(eigenenergies_total,o)
        plt.title(f"Plot of the overlap of probabilities Schmidt 1 and 2 in the total energy eigenbasis for EI={EI} and w={w}")
        plt.xlabel("Eigenenergies of H_total")
        plt.ylabel("Overlap in eigenstate i")
    else:
        plt.plot(eigenenergies_total,o)
        plt.title(f"Plot of the overlap of probabilities of Schmidt 1 and 2 in the total energy eigenbasis for EI={EI} and w={w}")
        plt.xlabel("Eigenenergies of H_total")
        plt.ylabel("Overlap in eigenstate i")
        plt.ylim(0, 0.002)
    
    # Add clock
    plt.text(0.95, 0.95, f"Frame: {frames}", horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes)

def gif_schmidt_overlap(eig,s_list, info_list, zoom=False): #EI,w,result,eigenstates_total,eigenenergies_total,env,d1,d2,E_spacing,tmax,ind_nb
    
    #Get the necessary information
    eigenstates_total=eig[1]
    eigenenergies_total=eig[0]
    ind_nb=info_list[10]
    s_full_list=s_list[3]
    
    # Create a figure
    fig = plt.figure(figsize=(10, 5))

    # Create the animation
    ani = FuncAnimation(fig, update,fargs=(eigenstates_total,eigenenergies_total,s_full_list,info_list,zoom), frames=ind_nb-1, interval=200)

    # Save the animation as a GIF
    path = f'../outputs/gifs/overlap_schmidts_param_{info_list}_zoom_{zoom}.gif'
    ani.save(path, writer='pillow')
    plt.close()

    return path





"""
def plot_p_overlap_graph_characterize(d1=10,d2=200,w=[0.1,0.2,0.3,0.4], E_spacing=1.0, EI=[0.03,0.05,0.07,0.09],tmax=30, ind_nb =100,env=[0]): #
    
    o01_list = []
    o02_list = []
    o12_list = []
    for wi in w:  
        for EIi in EI:
            print(f'wi_{wi}_EI_{EIi}')           
            result, tlist, H_q, H_system_2, H_system_1_ext, H_system_2_ext, H_interaction, H_total, ket_0, ket_1, initial_state_system_2 = generate_result(d1,d2,wi, E_spacing, EIi, tmax, ind_nb,1)
            eigenenergies_total, eigenstates_total = H_total.eigenstates()
            print("generated")
            o01 = []
            o02 = []
            o12 = []
            for idx in range(len(tlist)-1):
                s1=compute_schmidt_full(result,idx+1,1)
                s2=compute_schmidt_full(result,idx+1,2)
                global_state = result.states[idx+1]
                #s3=compute_schmidt_full(result,idx,3)
                o01.append(p_overlap(global_state,s1,eigenstates_total))
                o02.append(p_overlap(global_state,s2,eigenstates_total))
                o12.append(p_overlap(s1,s2,eigenstates_total))
            o01_list.append(o01)
            o02_list.append(o02)
            o12_list.append(o12)
        
    fig, axs = plt.subplots(len(w), len(EI), figsize=(10, 2*len(w)), sharex=True, sharey=True)
    #plt.title(f"Plot of the means and standard dev of the distributions of Schmidt 1 and 2 w={w}, EI={EI}, env={env}", fontsize=10)
    for i in range(len(w)):
        for j in range(len(EI)):
            axs[i, j].plot(tlist[0:len(tlist) - 1], o01_list[i*len(EI) + j])
            axs[i, j].plot(tlist[0:len(tlist) - 1], o02_list[i*len(EI) + j])
            axs[i, j].plot(tlist[0:len(tlist) - 1], o12_list[i*len(EI) + j])
            #axs[i, j].set_title(f"Plot of the means and standard dev of the distributions of Schmidt 1 and 2 w={w[i]}, EI={EI[j]}, env={env}", fontsize=5)
            axs[i, j].set_xlabel("Time", fontsize=6)
            axs[i, j].set_ylabel("Overlap", fontsize=8)
            mean = get_mean_rd_overlap(w[i],EI[j])
            axs[i, j].axhline(y=mean, color='red', linestyle='--')
            
    plt.legend(["o01", "o02", "012"])
    plt.tight_layout()
    plt.savefig(f'Graphs/overlap_characterization_EI_{EI},w_{w},env_{env}.png')
    plt.show()


#This is the piece of code I used to find the mean overlap between newly initialized eigenstates. But note that this is for a very specific set of parameters
def get_mean_rd_overlap(w = 0.3,Int_strength = 0.052):
    d1, d2 = 10, 200
    E_spacing = 1.0
    
    # Create basis states for system 1 and system 2
    basis_system_1 = [qt.basis(d1, i) for i in range(d1)]
    basis_system_2 = [qt.basis(d2, i) for i in range(d2)]
    ket_0 = qt.basis(d1, 3)  # |0> state
    ket_1 = qt.basis(d1, 7)  # |2> state, int(dim_system_1/2)
        
    # Define random Hermitian matrices as Hamiltonians for system 1 and system 2
    H_system_1 = qt.qeye(d1) #qt.rand_herm(dim_system_1)  # Random Hermitian matrix for system 1
    energy_spacing = E_spacing  # Adjust as needed
    diagonal_elements = np.arange(0, d1) * 1.0
    H_q = qt.Qobj(np.diag(diagonal_elements)) # Create a diagonal matrix with increasing diagonal elements
    H_system_2_1 = qt.rand_herm(d2,1)  # Random Hermitian matrix for system 2
    H_system_2_2 = qt.rand_herm(d2,1)  # Random Hermitian matrix for system 2
    H_system_2_3 = qt.rand_herm(d2,1)  # Random Hermitian matrix for system 2
    H_system_2_4 = qt.rand_herm(d2,1)  # Random Hermitian matrix for system 2
    H_system_2_5 = qt.rand_herm(d2,1)  # Random Hermitian matrix for system 2
    H_system_2_6 = qt.rand_herm(d2,1)  # Random Hermitian matrix for system 2
    H_system_2_7 = qt.rand_herm(d2,1)  # Random Hermitian matrix for system 2
    H_system_2_8 = qt.rand_herm(d2,1)  # Random Hermitian matrix for system 2
    H_system_2_9 = qt.rand_herm(d2,1)  # Random Hermitian matrix for system 2
    H_system_2_10 = qt.rand_herm(d2,1)  # Random Hermitian matrix for system 2
    # Define initial states for system 1 and system 2
    initial_state_system_1 = (math.sqrt(w)*ket_0 + math.sqrt(1-w)*ket_1).unit()
    #initial_state_system_2 = qt.rand_ket(dim_system_2)
    ev1 ,es1 = H_system_2_1.eigenstates()
    ev2 ,es2 = H_system_2_2.eigenstates()
    ev3 ,es3 = H_system_2_3.eigenstates()
    ev4 ,es4 = H_system_2_4.eigenstates()
    ev5 ,es5 = H_system_2_5.eigenstates()
    ev6 ,es6 = H_system_2_6.eigenstates()
    ev7 ,es7 = H_system_2_7.eigenstates()
    ev8 ,es8 = H_system_2_8.eigenstates()
    ev9 ,es9 = H_system_2_9.eigenstates()
    ev10,es10 = H_system_2_10.eigenstates()

    initial_state_system_2_1 = es1[round(d2/2)]
    initial_state_system_2_2 = es2[round(d2/2)]
    initial_state_system_2_3 = es3[round(d2/2)]
    initial_state_system_2_4 = es4[round(d2/2)]
    initial_state_system_2_5 = es5[round(d2/2)]
    initial_state_system_2_6 = es6[round(d2/2)]
    initial_state_system_2_7 = es7[round(d2/2)]
    initial_state_system_2_8 = es8[round(d2/2)]
    initial_state_system_2_9 = es9[round(d2/2)]
    initial_state_system_2_10 = es10[round(d2/2)]
    #define initial state of full system
    states=[]
    states.append(qt.tensor(initial_state_system_1, initial_state_system_2_1))
    states.append(qt.tensor(initial_state_system_1, initial_state_system_2_2))
    states.append(qt.tensor(initial_state_system_1, initial_state_system_2_3))
    states.append(qt.tensor(initial_state_system_1, initial_state_system_2_4))
    states.append(qt.tensor(initial_state_system_1, initial_state_system_2_5))
    states.append(qt.tensor(initial_state_system_1, initial_state_system_2_6))
    states.append(qt.tensor(initial_state_system_1, initial_state_system_2_7))
    states.append(qt.tensor(initial_state_system_1, initial_state_system_2_8))
    states.append(qt.tensor(initial_state_system_1, initial_state_system_2_9))
    states.append(qt.tensor(initial_state_system_1, initial_state_system_2_10))

    interaction_strength = Int_strength  # Adjust as needed
    H_interaction = interaction_strength * qt.tensor(H_q, qt.rand_herm(d2,1))  
        
    H_system_1_ext = qt.tensor(H_system_1, qt.qeye(d2))
    H_system_2_ext = 0.75*qt.tensor(qt.qeye(d1), H_system_2_1)
    H_total = H_system_1_ext + H_system_2_ext + H_interaction

    eigenenergies_total, eigenstates_total = H_total.eigenstates() 

    st=[]
    for s in states:
        st.append(s.full().squeeze())

    state_0=st[0]
    p_0=[abs(np.vdot(state_0, eigenstate)) for eigenstate in eigenstates_total]
    st.pop(0)

    overlap_list=[]
    for s in st:
        p = [abs(np.vdot(s, eigenstate)) for eigenstate in eigenstates_total]
        overlap_list.append(np.dot(p_0, p))

    mean_overlap = np.mean(overlap_list)
    return mean_overlap
"""