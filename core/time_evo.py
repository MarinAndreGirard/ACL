import numpy as np
import qutip as qt
import math
import matplotlib.pyplot as plt
import create_hamiltonian as ch
import create_hamiltonian as ch
import create_state as cs
import sys
import os

def time_evo(d1=10,d2=200,E_spacing = 1.0, E_int = 0.03, E_int2=0, E_env=1, E_env2=0,w=[0,0,0,np.sqrt(0.3),0,0,0,np.sqrt(0.7),0,0],envi=[0], tmax= 10, ind_nb = 100,log=0,E_s=0):
    """_summary_

    Args:
        d1 (int, optional): dimension of the system. Defaults to 10.
        d2 (int, optional): dimension of the environment. Defaults to 200.
        E_spacing (float, optional): energy between each level of the truncated simple harmonic oscillator. Defaults to 1.0.
        E_int (float, optional): interaction strength between the system and enviroment. Defaults to 0.03.
        E_int2 (int, optional): constant energy of the interaction term. Defaults to 0.
        E_env (int, optional): energy factor in front of environment self hamiltonian. Defaults to 1.
        E_env2 (int, optional): constant enerfy of the environemnt self interaction. Defaults to 0.
        envi (list, optional): list of probabilities of the initial state of the environment to be in environment self interaction energy eigenstates. Defaults to [0] which sets it to the d2/2 energy eigenstate.
        w (list, optional): list of probabilities of the initial state of the system to be in the SHO energy eigenstates. Defaults to [0,0,0,np.sqrt(0.3),0,0,0,np.sqrt(0.7),0,0].
        tmax (int, optional): max time of the time evolution. Defaults to 10.
        ind_nb (int, optional): number of time steps. Defaults to 100.
        log (int, optional): defines if steps are taken linearly (0) or logarithmically (1). Defaults to 0.

    Raises:
        ValueError: you need as many weights as there are dimensions in the system

    Returns:
        result, tlist, H_list, state_list: result (TODO), state_list (TODO), H_list (TODO), tlist: list of times at which the time evolution was calculated
    """

    #TODO:
    #- make sure I have all the cool features of q_solve before closing it forever
    #- finish docstring
    #- Do some testing comparing results from essolve and mesolve
    #- Make it output all the relevant information in an array so i can use it to title graphs.
    info_list=[d1,d2,E_spacing, E_int, E_int2, E_env, E_env2,w,envi,tmax,ind_nb,log,E_s]
    if len(w) != d1:
        raise ValueError("Length of 'w' and 'd1' must be the same")

    H_list = ch.create_H(d1,d2,E_spacing, E_int, E_int2, E_env, E_env2,E_s)
    
    H=H_list[1]
    state_list = cs.create_state(d1,d2,H_list[8],w,envi) 
     
    tlist = np.linspace(0, tmax, ind_nb) # Linear spacing
    if log == 0:
        tlist = np.linspace(0, tmax, ind_nb)  # Linear spacing
    elif log == 1:
        tlist = np.logspace(np.log10(1), np.log10(tmax+1), ind_nb)-1  # Logarithmic spacing
    else:
        raise ValueError("Invalid value for 'log'. It should be either 0 or 1.")
    info_list=[d1,d2,E_spacing, E_int, E_int2, E_env, E_env2,w,envi,tmax,ind_nb,log,E_s,tlist]
    # Perform time evolution of the combined system
    result = qt.mesolve(H, state_list[0], tlist, [], [])
    #result = qt.essolve(H, state_list[0], tlist, [], [])
    
    return result, tlist, H_list, state_list, info_list

def time_evo_new(d1=10,d2=200,E_s=1, E_s2=0, E_int_s=0.03, E_int_e=1,E_int_s2=0,E_int_e2=0, E_e=1, E_e2=0,w=[0,0,0,np.sqrt(0.3),0,0,0,np.sqrt(0.7),0,0],envi=[0], tmax= 10, ind_nb = 100,log=0,file_name="simulation_results.txt"):
    """_summary_ 
    Args:
        d1 (int, optional): dimension of the system. Defaults to 10.
        d2 (int, optional): dimension of the environment. Defaults to 200.
        E_spacing (float, optional): energy between each level of the truncated simple harmonic oscillator. Defaults to 1.0.
        E_int (float, optional): interaction strength between the system and enviroment. Defaults to 0.03.
        E_int2 (int, optional): constant energy of the interaction term. Defaults to 0.
        E_env (int, optional): energy factor in front of environment self hamiltonian. Defaults to 1.
        E_env2 (int, optional): constant enerfy of the environemnt self interaction. Defaults to 0.
        envi (list, optional): list of probabilities of the initial state of the environment to be in environment self interaction energy eigenstates. Defaults to [0] which sets it to the d2/2 energy eigenstate.
        w (list, optional): list of probabilities of the initial state of the system to be in the SHO energy eigenstates. Defaults to [0,0,0,np.sqrt(0.3),0,0,0,np.sqrt(0.7),0,0].
        tmax (int, optional): max time of the time evolution. Defaults to 10.
        ind_nb (int, optional): number of time steps. Defaults to 100.
        log (int, optional): defines if steps are taken linearly (0) or logarithmically (1). Defaults to 0.

    Raises:
        ValueError: you need as many weights as there are dimensions in the system

    Returns:
        result, tlist, H_list, state_list: result (TODO), state_list (TODO), H_list (TODO), tlist: list of times at which the time evolution was calculated
    """

    #TODO:
    #- make sure I have all the cool features of q_solve before closing it forever
    #- finish docstring
    #- Do some testing comparing results from essolve and mesolve
    #- Make it output all the relevant information in an array so i can use it to title graphs.
    if len(w) != d1:
        raise ValueError("Length of 'w' and 'd1' must be the same")

    #H_list = ch.create_H(d1,d2,E_spacing, E_int, E_int2, E_env, E_env2,E_s)
    H_list = ch.create_H_new(d1,d2, E_s, E_s2, E_int_s, E_int_e,E_int_s2, E_int_e2, E_e, E_e2)

    H=H_list[1]
    state_list = cs.create_state(d1,d2,H_list[8],w,envi) 
     
    tlist = np.linspace(0, tmax, ind_nb) # Linear spacing
    if log == 0:
        tlist = np.linspace(0, tmax, ind_nb)  # Linear spacing
    elif log == 1:
        tlist = np.logspace(np.log10(1), np.log10(tmax+1), ind_nb)-1  # Logarithmic spacing
    else:
        raise ValueError("Invalid value for 'log'. It should be either 0 or 1.")
    info_list=[d1,d2,E_s, E_s2, E_int_s, E_int_e,E_int_s2,E_int_e2, E_e, E_e2,w,envi, tmax, ind_nb,log,tlist] #TODO update info_list and make relevant changes to other functions
    
    # Perform time evolution of the combined system
    result = qt.mesolve(H, state_list[0], tlist, [], [])
    #result = qt.essolve(H, state_list[0], tlist, [], [])
    
    # Save outputs in a .txt file
    outputs_dir = "outputs/simulation_results"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    # Save parameters in a .txt file
    params_file_path = os.path.join(outputs_dir, "params_" + file_name)
    with open(params_file_path, "w") as f:
        f.write(f"d1 === {d1}\n")
        f.write(f"d2 === {d2}\n")
        f.write(f"E_s === {E_s}\n")
        f.write(f"E_s2 === {E_s2}\n")
        f.write(f"E_int_s === {E_int_s}\n")
        f.write(f"E_int_e === {E_int_e}\n")
        f.write(f"E_int_s2 === {E_int_s2}\n")
        f.write(f"E_int_e2 === {E_int_e2}\n")
        f.write(f"E_e === {E_e}\n")
        f.write(f"E_e2 === {E_e2}\n")
        f.write(f"w === {w}\n")
        f.write(f"envi === {envi}\n")
        f.write(f"tmax === {tmax}\n")
        f.write(f"ind_nb === {ind_nb}\n")
        f.write(f"log === {log}\n")
    
    # Save parameters in a .txt file
    tlist_file_path = os.path.join(outputs_dir, "tlist_" + file_name)
    np.save(tlist_file_path, tlist)
    #with open(tlist_file_path, "w") as f:
    #   f.write(f"{tlist}")

    # Save result in a .txt file
    result_file_path = os.path.join(outputs_dir, "result_" + file_name)
    qt.qsave(result, result_file_path)
    
    # Save H_list in a .txt file
    
    H_list_file_path = os.path.join(outputs_dir, "H_list_" + file_name)
    #H_total, H_s, H_int, H_e, H_s_self, H_int_s, H_int_e, H_e_self
    
    qt.qsave(H_list, H_list_file_path)
    
    return result, tlist, H_list, state_list, info_list

def time_evo_from_state(state_s,d1=10,d2=200,E_s=1, E_s2=0, E_int_s=0.03, E_int_e=1,E_int_s2=0,E_int_e2=0, E_e=1, E_e2=0,envi=[0], tmax= 10, ind_nb = 100,log=0,file_name="simulation_results.txt"):
    """_summary_ 
    Args:
        d1 (int, optional): dimension of the system. Defaults to 10.
        d2 (int, optional): dimension of the environment. Defaults to 200.
        E_spacing (float, optional): energy between each level of the truncated simple harmonic oscillator. Defaults to 1.0.
        E_int (float, optional): interaction strength between the system and enviroment. Defaults to 0.03.
        E_int2 (int, optional): constant energy of the interaction term. Defaults to 0.
        E_env (int, optional): energy factor in front of environment self hamiltonian. Defaults to 1.
        E_env2 (int, optional): constant enerfy of the environemnt self interaction. Defaults to 0.
        envi (list, optional): list of probabilities of the initial state of the environment to be in environment self interaction energy eigenstates. Defaults to [0] which sets it to the d2/2 energy eigenstate.
        w (list, optional): list of probabilities of the initial state of the system to be in the SHO energy eigenstates. Defaults to [0,0,0,np.sqrt(0.3),0,0,0,np.sqrt(0.7),0,0].
        tmax (int, optional): max time of the time evolution. Defaults to 10.
        ind_nb (int, optional): number of time steps. Defaults to 100.
        log (int, optional): defines if steps are taken linearly (0) or logarithmically (1). Defaults to 0.

    Raises:
        ValueError: you need as many weights as there are dimensions in the system

    Returns:
        result, tlist, H_list, state_list: result (TODO), state_list (TODO), H_list (TODO), tlist: list of times at which the time evolution was calculated
    """

    #TODO:
    # Merge this function with time_evo_new somehow.
    
    #H_list = ch.create_H(d1,d2,E_spacing, E_int, E_int2, E_env, E_env2,E_s)
    H_list = ch.create_H_new(d1,d2, E_s, E_s2, E_int_s, E_int_e,E_int_s2, E_int_e2, E_e, E_e2)

    H=H_list[1]
     
    H_e_self=H_list[8]
    ev ,es = H_e_self.eigenstates()
    if np.array_equal(envi, [0]):
        state_e = es[round(d2/2)] #Define initial state of environment case 1
    else:
        l = len(envi)
        if l != d2:
            raise ValueError("Length of 'envi' and 'd2' must be the same")
        state_e = sum([envi[i]*es[i] for i in range(len(ket_list))]).unit() #Define initial state of environment case 2

    state = qt.tensor(state_s, state_e)

    tlist = np.linspace(0, tmax, ind_nb) # Linear spacing
    if log == 0:
        tlist = np.linspace(0, tmax, ind_nb)  # Linear spacing
    elif log == 1:
        tlist = np.logspace(np.log10(1), np.log10(tmax+1), ind_nb)-1  # Logarithmic spacing
    else:
        raise ValueError("Invalid value for 'log'. It should be either 0 or 1.")
    info_list=[d1,d2,E_s, E_s2, E_int_s, E_int_e,E_int_s2,E_int_e2, E_e, E_e2,envi,tmax, ind_nb,log,tlist] #TODO update info_list and make relevant changes to other functions
    
    # Perform time evolution of the combined system
    result = qt.mesolve(H, state, tlist, [], [])

    # Save outputs in a .txt file
    outputs_dir = "outputs/simulation_results"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    # Save parameters in a .txt file
    params_file_path = os.path.join(outputs_dir, "params_" + file_name)
    with open(params_file_path, "w") as f:
        f.write(f"d1 === {d1}\n")
        f.write(f"d2 === {d2}\n")
        f.write(f"E_s === {E_s}\n")
        f.write(f"E_s2 === {E_s2}\n")
        f.write(f"E_int_s === {E_int_s}\n")
        f.write(f"E_int_e === {E_int_e}\n")
        f.write(f"E_int_s2 === {E_int_s2}\n")
        f.write(f"E_int_e2 === {E_int_e2}\n")
        f.write(f"E_e === {E_e}\n")
        f.write(f"E_e2 === {E_e2}\n")
        f.write(f"envi === {envi}\n")
        f.write(f"tmax === {tmax}\n")
        f.write(f"ind_nb === {ind_nb}\n")
        f.write(f"log === {log}\n")
    
    # Save parameters in a .txt file
    tlist_file_path = os.path.join(outputs_dir, "tlist_" + file_name)
    np.save(tlist_file_path, tlist)
    #with open(tlist_file_path, "w") as f:
    #   f.write(f"{tlist}")

    # Save result in a .txt file
    result_file_path = os.path.join(outputs_dir, "result_" + file_name)
    qt.qsave(result, result_file_path)
    
    # Save H_list in a .txt file
    
    H_list_file_path = os.path.join(outputs_dir, "H_list_" + file_name)
    #H_total, H_s, H_int, H_e, H_s_self, H_int_s, H_int_e, H_e_self
    
    qt.qsave(H_list, H_list_file_path)
    
    return result, tlist, H_list, state, info_list




def load_param(file_name):
    """
    Load the results saved in a file by the `time_evo_new` function.
    
    Args:
        file_name (str): path to the output file.
        
    Returns:
        d1, d2, E_s, E_s2, E_int_s, E_int_e, E_int_s2, E_int_e2, E_e, E_e2, w, envi, tmax, ind_nb, log, tlist, result, H_list, state_list, info_list: variables recovered from the file.
    """
    #outputs_dir = "outputs/simulation_results"
    outputs_dir = "outputs/simulation_results"
    params_file_path = os.path.join(outputs_dir, "params_" + file_name)
    #result_file_path = os.path.join(outputs_dir, "result_" + file_name)
    #H_list_file_path = os.path.join(outputs_dir, "H_list_" + file_name)
    #state_list_file_path = os.path.join(outputs_dir, "state_list_" + file_name)
    #info_list_file_path = os.path.join(outputs_dir, "info_list_" + file_name)
    
    with open(params_file_path, "r") as f:
        lines = f.readlines()
    
    # Extract parameters
    params = {}
    for line in lines:
        param, value = line.strip().split(" === ")
        params[param.strip()] = eval(value.strip())
    
    # Extract results
    #with open(result_file_path, "r") as f:
    #    result = eval(f.readline().strip().split(" === ")[1])
    #with open(H_list_file_path, "r") as f:
    #    H_list = eval(f.readline().strip().split(" === ")[1])
    #with open(state_list_file_path, "r") as f:
    #    state_list = eval(f.readline().strip().split(" === ")[1])
    #with open(info_list_file_path, "r") as f:
    #    info_list = eval(f.readline().strip().split(" === ")[1])
    
    # Recover variables
    #d1, d2, E_s, E_s2, E_int_s, E_int_e, E_int_s2, E_int_e2, E_e, E_e2, w, envi, tmax, ind_nb, log= params.values()
    
    return params.values()

def load_tlist(file_name):

    outputs_dir = "outputs/simulation_results"
    tlist_file_path = os.path.join(outputs_dir, "tlist_" + file_name + ".npy")
    tlist_temp=np.load(tlist_file_path)

    return tlist_temp

def load_result(file_name):
    outputs_dir = "outputs/simulation_results"
    result_file_path = os.path.join(outputs_dir, "result_" + file_name)
    r = qt.qload(result_file_path)
    return r

def load_H_list(file_name):
    outputs_dir = "outputs/simulation_results"
    H_list_file_path = os.path.join(outputs_dir, "H_list_" + file_name)
    r = qt.qload(H_list_file_path)
    return r


def time_evo_new_system_eig(d1=10,d2=200,E_s=1, E_s2=0, E_int_s=0.03, E_int_e=1,E_int_s2=0,E_int_e2=0, E_e=1, E_e2=0,w=[0,0,0,np.sqrt(0.3),0,0,0,np.sqrt(0.7),0,0],envi=[0], tmax= 10, ind_nb = 100,log=0,file_name="simulation_results.txt"):
    """_summary_ 
    Args:
        d1 (int, optional): dimension of the system. Defaults to 10.
        d2 (int, optional): dimension of the environment. Defaults to 200.
        E_spacing (float, optional): energy between each level of the truncated simple harmonic oscillator. Defaults to 1.0.
        E_int (float, optional): interaction strength between the system and enviroment. Defaults to 0.03.
        E_int2 (int, optional): constant energy of the interaction term. Defaults to 0.
        E_env (int, optional): energy factor in front of environment self hamiltonian. Defaults to 1.
        E_env2 (int, optional): constant enerfy of the environemnt self interaction. Defaults to 0.
        envi (list, optional): list of probabilities of the initial state of the environment to be in environment self interaction energy eigenstates. Defaults to [0] which sets it to the d2/2 energy eigenstate.
        w (list, optional): list of probabilities of the initial state of the system to be in the SHO energy eigenstates. Defaults to [0,0,0,np.sqrt(0.3),0,0,0,np.sqrt(0.7),0,0].
        tmax (int, optional): max time of the time evolution. Defaults to 10.
        ind_nb (int, optional): number of time steps. Defaults to 100.
        log (int, optional): defines if steps are taken linearly (0) or logarithmically (1). Defaults to 0.

    Raises:
        ValueError: you need as many weights as there are dimensions in the system

    Returns:
        result, tlist, H_list, state_list: result (TODO), state_list (TODO), H_list (TODO), tlist: list of times at which the time evolution was calculated
    """

    #TODO:
    #- make sure I have all the cool features of q_solve before closing it forever
    #- finish docstring
    #- Do some testing comparing results from essolve and mesolve
    #- Make it output all the relevant information in an array so i can use it to title graphs.
    if len(w) != d1:
        raise ValueError("Length of 'w' and 'd1' must be the same")

    #H_list = ch.create_H(d1,d2,E_spacing, E_int, E_int2, E_env, E_env2,E_s)
    H_list = ch.create_H_new(d1,d2, E_s, E_s2, E_int_s, E_int_e,E_int_s2, E_int_e2, E_e, E_e2)

    H=H_list[1]
    ket_list = [qt.basis(d1, i) for i in range(d1)] #Define the basis states of the system

    s_eigenenergies,s_eigenstates=H_list[6].eigenstates()
    state_s = s_eigenstates[10]#sum([w[i]*s_eigenstates[i] for i in range(len(s_eigenstates))]).unit() #Define the initial state of the system

    ev ,es = H_list[8].eigenstates()
    if np.array_equal(envi, [0]):
        state_e = es[round(d2/2)] #Define initial state of environment case 1
    else:
        l = len(envi)
        if l != d2:
            raise ValueError("Length of 'envi' and 'd2' must be the same")
        state_e = sum([envi[i]*es[i] for i in range(len(ket_list))]).unit() #Define initial state of environment case 2

    #define initial state of full system
    state = qt.tensor(state_s, state_e)

    state_list=[state,ket_list]
    
     
    tlist = np.linspace(0, tmax, ind_nb) # Linear spacing
    if log == 0:
        tlist = np.linspace(0, tmax, ind_nb)  # Linear spacing
    elif log == 1:
        tlist = np.logspace(np.log10(1), np.log10(tmax+1), ind_nb)-1  # Logarithmic spacing
    else:
        raise ValueError("Invalid value for 'log'. It should be either 0 or 1.")
    info_list=[d1,d2,E_s, E_s2, E_int_s, E_int_e,E_int_s2,E_int_e2, E_e, E_e2,w,envi, tmax, ind_nb,log,tlist] #TODO update info_list and make relevant changes to other functions
    
    # Perform time evolution of the combined system
    result = qt.mesolve(H, state_list[0], tlist, [], [])
    #result = qt.essolve(H, state_list[0], tlist, [], [])
    
    # Save outputs in a .txt file
    outputs_dir = "outputs/simulation_results"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    
    # Save parameters in a .txt file
    params_file_path = os.path.join(outputs_dir, "params_" + file_name)
    with open(params_file_path, "w") as f:
        f.write(f"d1 === {d1}\n")
        f.write(f"d2 === {d2}\n")
        f.write(f"E_s === {E_s}\n")
        f.write(f"E_s2 === {E_s2}\n")
        f.write(f"E_int_s === {E_int_s}\n")
        f.write(f"E_int_e === {E_int_e}\n")
        f.write(f"E_int_s2 === {E_int_s2}\n")
        f.write(f"E_int_e2 === {E_int_e2}\n")
        f.write(f"E_e === {E_e}\n")
        f.write(f"E_e2 === {E_e2}\n")
        f.write(f"w === {w}\n")
        f.write(f"envi === {envi}\n")
        f.write(f"tmax === {tmax}\n")
        f.write(f"ind_nb === {ind_nb}\n")
        f.write(f"log === {log}\n")
    
    # Save parameters in a .txt file
    tlist_file_path = os.path.join(outputs_dir, "tlist_" + file_name)
    np.save(tlist_file_path, tlist)
    #with open(tlist_file_path, "w") as f:
    #   f.write(f"{tlist}")

    # Save result in a .txt file
    result_file_path = os.path.join(outputs_dir, "result_" + file_name)
    qt.qsave(result, result_file_path)
    
    # Save H_list in a .txt file
    
    H_list_file_path = os.path.join(outputs_dir, "H_list_" + file_name)
    #H_total, H_s, H_int, H_e, H_s_self, H_int_s, H_int_e, H_e_self
    
    qt.qsave(H_list, H_list_file_path)
    
    return result, tlist, H_list, state_list, info_list