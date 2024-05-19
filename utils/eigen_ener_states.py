

def eigen_ener_states(H_list):
    """_summary_

    Args:
        H_s (Qobj): _description_
        H_e (Qobj): _description_
        H (Qobj): _description_
        H_s_self (Qobj): _description_
    """
    H_e=H_list[4]
    H=H_list[1]
    H_int_s=H_list[6]

    # Calculate the eigenenergies and eigenstates of the Hamiltonians  
    eigenenergies_system_2, eigenstates_system_2 = H_e.eigenstates() 
    eigenenergies_system_total, eigenstates_system_total = H.eigenstates() 
    eigenenergies_system_1, eigenstates_system_1 = H_int_s.eigenstates() 

    return eigenenergies_system_1, eigenenergies_system_2, eigenenergies_system_total, eigenstates_system_1, eigenstates_system_2, eigenstates_system_total