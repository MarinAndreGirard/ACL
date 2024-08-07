import qutip as qt
import numpy as np

# this file will contain funciton for returning the expectation values of the various Hamiltonians in time.



def exp_val(state, H_list):
    exp_val_time = []

    E_tot = np.vdot(state.full(), np.dot(H_list[1].full(), state.full()))
    E_s = np.vdot(state.full(), np.dot(H_list[2].full(), state.full()))
    E_int = np.vdot(state.full(), np.dot(H_list[3].full(), state.full()))
    E_e = np.vdot(state.full(), np.dot(H_list[4].full(), state.full()))

    exp_val_time = [E_tot, E_s, E_int, E_e]        
    return exp_val_time

def exp_val_time(result, H_list, tlist):
    #TODO make this call exp_val
    exp_val_time = []
    E_tot=[]
    E_s=[]
    E_int=[]
    E_e=[]

    for i in range(len(tlist)):
        E_tot.append(qt.expect(H_list[1],result.states[i]))
        E_s.append(qt.expect(H_list[2],result.states[i]))
        E_int.append(qt.expect(H_list[3],result.states[i]))
        E_e.append(qt.expect(H_list[4],result.states[i]))

    exp_val_time = [E_tot, E_s, E_int, E_e]        
    return exp_val_time

def exp_val_time_file(result, H_list, info, file_name):
    tlist = info[15]
    exp_val_time = []
    E_tot=[]
    E_s=[]
    E_int=[]
    E_e=[]

    for i in range(len(tlist)):
        E_tot.append(qt.expect(H_list[1],result.states[i]))
        E_s.append(qt.expect(H_list[2],result.states[i]))
        E_int.append(qt.expect(H_list[3],result.states[i]))
        E_e.append(qt.expect(H_list[4],result.states[i]))

    exp_val_time = [E_tot, E_s, E_int, E_e]     

    with open(file_name, 'w') as f:
        for t, val in zip(tlist, exp_val_time):
            f.write(str(t) + ' ' + ' '.join([str(v) for v in val]) + '\n')
