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

def load_outputs_3(file_name):
    """
    Load the results saved in a file by the `time_evo_new` function.
    
    Args:
        file_name (str): path to the output file.
        
    Returns:
        tuple: result, tlist, H_list, state_list, info_list
    """
    outputs_dir = "outputs/simulation_results"
    file_path = os.path.join(outputs_dir, file_name)
    with open(file_path, "r") as f:
        lines = f.readlines()
    params = {}
    for line in lines[1:12]:
        k, v = line.strip().split("=")
        params[k.strip()] = eval(v.strip())
    results = []
    for line in lines[12:]:
        results.append(eval(line.strip()))
    return results[0], results[1], results[2], results[3], params

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
    
    # Save outputs in a .txt file in the same directory as the function was called
    outputs_dir = "outputs/simulation_results"
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    file_path = os.path.join(outputs_dir, file_name)
    with open(file_path, "w") as f:
        f.write("Parameters taken by the function:\n")
        for k,v in locals().items():
            f.write(f"{k} = {v}\n")
        f.write("Results:\n")
        f.write(f"result = {result}\n")
        f.write(f"H_list = {H_list}\n")
        f.write(f"state_list = {state_list}\n")
        f.write(f"info_list = {info_list}\n")
    
    return result, tlist, H_list, state_list, info_list


def load_outputs(file_name):
    """
    Load the results saved in a file by the `time_evo_new` function.
    
    Args:
        file_name (str): path to the output file.
        
    Returns:
        d1, d2, E_s, E_s2, E_int_s, E_int_e, E_int_s2, E_int_e2, E_e, E_e2, w, envi, tmax, ind_nb, log, tlist, result, H_list, state_list, info_list: variables recovered from the file.
    """
    with open(file_name, "r") as f:
        lines = f.readlines()
    
    # Extract parameters
    params = {}
    for line in lines[1:12]:
        param, value = line.split(" = ")
        params[param.strip()] = eval(value.strip())
    
    # Extract results
    results = {}
    for line in lines[13:]:
        key, value = line.split(" = ")
        results[key.strip()] = eval(value.strip())
    
    # Recover variables
    d1, d2, E_s, E_s2, E_int_s, E_int_e, E_int_s2, E_int_e2, E_e, E_e2, w, envi, tmax, ind_nb, log, tlist = params.values()
    result, H_list, state_list, info_list = results.values()
    
    return d1, d2, E_s, E_s2, E_int_s, E_int_e, E_int_s2, E_int_e2, E_e, E_e2, w, envi, tmax, ind_nb, log, tlist, result, H_list, state_list, info_list


def load_outputs_2(file_name):
    """
    Load the results saved in a file by the `time_evo_new` function.
    
    Args:
        file_name (str): path to the output file.
        
    Returns:
        d1, d2, E_s, E_s2, E_int_s, E_int_e, E_int_s2, E_int_e2, E_e, E_e2, w, envi, tmax, ind_nb, log, tlist, result, H_list, state_list, info_list: variables recovered from the file.
    """
    with open(file_name, "r") as f:
        lines = f.readlines()
    
    # Extract parameters
    params = {line.split(" = ")[0]: eval(line.split(" = ")[1]) for line in lines[1:12]}
    
    # Extract results
    results = {line.split(" = ")[0]: eval(line.split(" = ")[1]) for line in lines[13:]}
    
    # Recover variables
    d1, d2, E_s, E_s2, E_int_s, E_int_e, E_int_s2, E_int_e2, E_e, E_e2, w, envi, tmax, ind_nb, log, tlist = params.values()
    result, H_list, state_list, info_list = results.values()
    
    return d1, d2, E_s, E_s2, E_int_s, E_int_e, E_int_s2, E_int_e2, E_e, E_e2, w, envi, tmax, ind_nb, log, tlist, result, H_list, state_list, info_list


