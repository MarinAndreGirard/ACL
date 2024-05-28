

def eigen_ener_states(H_list):
    """_summary_

    Args:
        H_s (Qobj): _description_
        H_e (Qobj): _description_
        H (Qobj): _description_
        H_s_self (Qobj): _description_

    Returns: eigenenergies_total,eigenstates_total,eigenenergies_s,eigenstates_s,eigenenergies_int,eigenstates_int,eigenenergies_e,eigenstates_e,eigenenergies_s_self,eigenstates_s_self,eigenenergies_int_s,eigenstates_int_s,eigenenergies_int_e,eigenstates_int_e,eigenenergies_e_self,eigenstates_e_self
    The nomenclature is the same as for the Hamiltonians.
                    """



    #H_list: d, H_total, H_s, H_int, H_e, H_s_self, H_int_s, H_int_e, H_e_self

    eigenenergies_total,eigenstates_total =H_list[1].eigenstates()
    eigenenergies_s,eigenstates_s =H_list[2].eigenstates()
    eigenenergies_int,eigenstates_int =H_list[3].eigenstates()
    eigenenergies_e,eigenstates_e =H_list[4].eigenstates()
    eigenenergies_s_self,eigenstates_s_self =H_list[5].eigenstates()
    eigenenergies_int_s,eigenstates_int_s =H_list[6].eigenstates()
    eigenenergies_int_e,eigenstates_int_e =H_list[7].eigenstates()
    eigenenergies_e_self,eigenstates_e_self =H_list[8].eigenstates()

    # Calculate the eigenenergies and eigenstates of the Hamiltonians  

    eigenenergies_system_2, eigenstates_system_2 = H_list[4].eigenstates() #H_e
    eigenenergies_system_total, eigenstates_system_total = H_list[1].eigenstates() #H
    eigenenergies_system_1, eigenstates_system_1 = H_list[6].eigenstates() #H_int_s
    eigenenergies_env_1, eigenstates_env_1 = H_list[8].eigenstates()

    return eigenenergies_total,eigenstates_total,eigenenergies_s,eigenstates_s,eigenenergies_int,eigenstates_int,eigenenergies_e,eigenstates_e,eigenenergies_s_self,eigenstates_s_self,eigenenergies_int_s,eigenstates_int_s,eigenenergies_int_e,eigenstates_int_e,eigenenergies_e_self,eigenstates_e_self

#eigenenergies_system_2, eigenstates_system_2 = H_list[4].eigenstates() #H_e
 #   eigenenergies_system_total, eigenstates_system_total = H_list[1].eigenstates() #H
  #  eigenenergies_system_1, eigenstates_system_1 = H_list[6].eigenstates() #H_int_s
   # eigenenergies_env_1, eigenstates_env_1 = H_list[8].eigenstates()

#    return eigenenergies_system_1, eigenenergies_system_2, eigenenergies_system_total, eigenstates_system_1, eigenstates_system_2, eigenstates_system_total, eigenenergies_env_1, eigenstates_env_1