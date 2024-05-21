import numpy as np
import qutip as qt
import math

def create_state(d1,d2, H_e_self,w,envi=[0]):
        
    #TODO:
    #-Find a way to use a randome key to have consistent results and re-use them for testing.
    #-Add option to strt in non product state
    #-Add option of starting in not an environment self interaction eigenstate (or superposition of them)

    ket_list = [qt.basis(d1, i) for i in range(d1)]
    print(len(ket_list))
    print(len(w))
    state_s = sum([w[i]*ket_list[i] for i in range(len(ket_list))]).unit()

    ev ,es = H_e_self.eigenstates()
    if np.array_equal(envi, [0]):
        state_e = es[round(d2/2)]
    else:
        l = len(envi)
        if l != d2:
            raise ValueError("Length of 'envi' and 'd2' must be the same")
        state_e = sum([envi[i]*es[i] for i in range(len(ket_list))]).unit()

    #define initial state of full system
    state = qt.tensor(state_s, state_e)

    return state, ket_list
