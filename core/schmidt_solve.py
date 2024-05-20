import numpy as np
import qutip as qt
import math
import matplotlib.pyplot as plt
#TODO do i need all of theses packages?
#TODO go through and clean
#TODO implement the simpler method?

def compute_schmidt_states_new(result, time_index):
    global_state = result.states[time_index]
    density_matrix = qt.ptrace(global_state, [0]) # Calculate the density matrix at the specified time
    eigenvalues, eigenstates = density_matrix.eigenstates() # Compute the eigenstates and eigenvalues of the density matrix
    eigenstates = [np.array(state) for state in eigenstates]
    schmidt_states_s = []
    schmidt_states_e = []
    schmidt_values = []
    i=0
    for state, eigenvalue in zip(eigenstates, eigenvalues):
        schmidt_values.append(eigenvalue)
        if eigenvalue < 10e-14:
            # If the eigenvalue is zero, set the Schmidt state to a zero vector
            schmidt_states_s.append(np.zeros_like(state))
        else:
            #print(f"state {state}")
            i=i+1
            N=abs(np.vdot(state,state))
            schmidt_states_s.append(state/np.sqrt(N)) # Normalize

    # Sort the Schmidt states by eigenvalue in descending order
    schmidt_states_s, schmidt_values = zip(*sorted(zip(schmidt_states_s, schmidt_values), key=lambda x: -x[1]))
    d=np.size(global_state)
    d1 = np.size(schmidt_states_s[0])
    d2=d//d1
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
        #step 2:We have |phi_i>|a_i>. We know |phi_i> and want |a_i>. |phi_i>|a_i> = (phi^1_i|a_i>,phi^2_i|a_i>, ...)
        #find the first nonzero coeff of |phi_i> and use it to extract |a_i>
        nonzero_index = np.nonzero(state)[0][0] #will need to know the index of a nonzero value in |phi_i>
        #now extract a d2 sized vector from thatd2*d1 sized vector
        temp2 = temp[nonzero_index*d2:(nonzero_index+1)*d2] #for k the nnzero index, this is phi^k_i|a_i>
        temp3=temp2/state[nonzero_index]
        N = abs(np.vdot(temp3,temp3))
        
        schmidt_states_e.append(temp3/np.sqrt(N))

    return schmidt_states_s,schmidt_states_e,schmidt_values


#Note, outer(a,b) is (a1.b, a2.b,...) and |a>|b>=(a1|b>,a2|b>,...)
def compute_schmidt_full(result,idx,s=1):
    ss, se, sv = compute_schmidt_states_new(result, idx)
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



def compute_schmidt_states(result, time_index, subsystem_index=0, trigger=0):
    density_matrix = qt.ptrace(result.states[time_index], [subsystem_index]) # Calculate the density matrix at the specified time
    eigenvalues, eigenstates = density_matrix.eigenstates() # Compute the eigenstates and eigenvalues of the density matrix
    eigenstates = [np.array(state) for state in eigenstates]
    schmidt_states = []
    schmidt_values = []
    for state, eigenvalue in zip(eigenstates, eigenvalues):
        schmidt_values.append(eigenvalue)
        if eigenvalue == 0:
            # If the eigenvalue is zero, set the Schmidt state to a zero vector
            schmidt_states.append(np.zeros_like(state))
        else:
            schmidt_states.append(state / np.linalg.norm(state)) # Normalize
            #schmidt_states.append(state / np.sqrt(np.array(eigenvalue)))  # Normalize
    # Sort the Schmidt states by eigenvalue in descending order
    schmidt_states, schmidt_values = zip(*sorted(zip(schmidt_states, schmidt_values), key=lambda x: -x[1]))
    if trigger == 0:
        return schmidt_states
    else:
        return schmidt_values
    
def compute_schmidt_states_all_time(result, ind_nb):
    """_summary_

    Args:
        result (_type_): _description_
        ind_nb (int): number of time indices
        subsystem_index (int, optional): TODO. Defaults to 0.
        trigger (int, optional): TODO. Defaults to 0.

    Returns:
        _type_: returns a time list of the Schmidt states of the system (TODO allow for env Schmidt states?). The Schmidt states are sorted by eigenvalue in descending order. Also returns a time list of the eigenvalues of the Schmidt states.
        For example print(s[0][10][0]) returns the first Schmidt state of the system at time index 10. print(s[0][10][5]) returns the 6th Schmidt state of the system at time index 10. print(s[1][10][3]) returns the 4th Schmdit value (decreasing value) of the system at time index 10.
    
    """
    density_matrix = [qt.ptrace(result.states[time_index], [0]) for time_index in range(ind_nb)] # Calculate the density matrix at all time
    
    schmidt_states_tlist = []
    schmidt_values_tlist = []
    
    for time_index in range(ind_nb):
        eigenvalues, eigenstates = density_matrix[time_index].eigenstates() # Compute the eigenstates and eigenvalues of the density matrix
        eigenstates = [np.array(state) for state in eigenstates]

        schmidt_states = []
        schmidt_values = []
        
        for state, eigenvalue in zip(eigenstates, eigenvalues):
            schmidt_values.append(eigenvalue)
            if eigenvalue == 0:
                # If the eigenvalue is zero, set the Schmidt state to a zero vector
                schmidt_states.append(np.zeros_like(state))
            else:
                schmidt_states.append(state / np.linalg.norm(state)) # Normalize
                #schmidt_states.append(state / np.sqrt(np.array(eigenvalue)))  # Normalize
        # Sort the Schmidt states by eigenvalue in descending order
        schmidt_states, schmidt_values = zip(*sorted(zip(schmidt_states, schmidt_values), key=lambda x: -x[1]))
        schmidt_states_tlist.append(schmidt_states)
        schmidt_values_tlist.append(schmidt_values)

    return schmidt_states_tlist,schmidt_values_tlist


def plot_schmidt_value_time(result,tlist):

    s_val_0=[]
    s_val_1=[]
    t_ind = range(len(tlist))
    for idx in t_ind:
        ss, se, sv = compute_schmidt_states_new(result, idx)
        s_val_0.append(sv[0])
        s_val_1.append(sv[1])

    plt.figure(figsize=(10, 5))
    plt.plot(tlist,s_val_0, marker='+')
    plt.plot(tlist,s_val_1, marker='+')
    plt.title('Evolution of Top 2 Eigenvalues Over Time')
    plt.xlabel('Time')
    plt.ylabel('Eigenvalue Magnitude')

