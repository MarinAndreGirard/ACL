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

def compute_schmidt_states_all_time(result, ind_nb):
    """_summary_

    Args:
        result (_type_): _description_
        ind_nb (int): number of time indices
        subsystem_index (int, optional): TODO. Defaults to 0.
        trigger (int, optional): TODO. Defaults to 0.

    Returns:
        _type_: returns a time list of the Schmidt states of the system (TODO allow for env Schmidt states?). The Schmidt states are sorted by eigenvalue in descending order. Also returns a time list of the eigenvalues of the Schmidt states.
        first [] gives state of system for 0 of environment for 1 and associated eigenvalue for 2. second [] gives the time, third [] gives which Schmidt state, the 0'th 1st, ...
        For example print(s[0][10][0]) returns the first Schmidt state of the system at time index 10. print(s[0][10][5]) returns the 6th Schmidt state of the system at time index 10. print(s[1][10][3]) returns the 4th Schmdit state of the environment at time index 10.
        print(s[2][10][3]) returns the 4th Schmdit value (decreasing value) of the environment at time index 10

    """

    #TODO
    #-Another thing i want to do here is to outpute for all time the compute_schmidt_full() without redundancy in the computation
    #-Make it do both env and syst.
    schmidt_states_s_tlist=[]
    schmidt_states_e_tlist=[]
    schmidt_values_tlist=[]
    schmidt_full_tlist=[]


    for time_index in range(ind_nb):
        ss, se, sv = compute_schmidt_states_new(result, time_index)
        
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


    #full = compute_full_schmidt(ind_nb,schmidt_states_s_tlist,schmidt_states_e_tlist)

    #Here do what is dont in compute_schmidt_full() for all time
    

    return schmidt_states_s_tlist,schmidt_states_e_tlist,schmidt_values_tlist,schmidt_full_tlist



