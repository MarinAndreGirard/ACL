import numpy as np
import qutip as qt
import math

def create_state(d1,d2, H_e_self,w):
    
    #TODO figure out in which basis are tyeh kets taken? seems to be same basis as H_s_self, or the one in which we def id.
    #TODO find a way to use a randome key to have consistent results and re-use them for testing.
    #We give the choice of how which state superposition to use. and how many (not more than d1)
    #we give the choice of what enviroment state, ie can choose a random state, a env self int eigenstate or interaction eig, or total Hamiltonian  env eig
    #Do we superimpose pointer states? They depend on the system self interaction? 
    #Do we add the option of not starting in a product state? TODO DO ALL OF THAT

    #d1 = 10,d2 = 200,w = 0.3, E_spacing = 1.0, Int_strength = 0.03, tmax= 10, ind_nb = 100,log=0,k0=2, k1=7
    #d1 = 10,d2 = 200,w = 0.3, E_spacing = 1.0, Int_strength = 0.03, tmax= 10, ind_nb = 100,log=0,envi=[1,2]
    #TODO use these things above to make the function more general. ie, go see the other generate_result in the q_solve.py.

    #ket_0 = qt.basis(d1, round(d1*1/4))  # |0> state
    #ket_1 = qt.basis(d1, round(d1*3/4))  # |2> state, int(dim_system_1/2)
    #ket_list = [ket_0, ket_1]
    # Create basis states for system 1 and system 2
    #basis_system_1 = [qt.basis(d1, i) for i in range(d1)]
    #basis_system_2 = [qt.basis(d2, i) for i in range(d2)]

    ket_list = [qt.basis(d1, i) for i in range(d1)]
    print(len(ket_list))
    print(len(w))
    state_s = sum([w[i]*ket_list[i] for i in range(len(ket_list))]).unit()

    #initial_state_system_2 = qt.rand_ket(dim_system_2)
    ev ,es = H_e_self.eigenstates()
    state_e = es[round(d2/2)]#TODO is there a better way to do this
    #define initial state of full system
    state = qt.tensor(state_s, state_e)

    return state, ket_list
    