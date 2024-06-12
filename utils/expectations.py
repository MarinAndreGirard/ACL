



# this file will contain funciton for returning the expectation values of the various Hamiltonians in time.


def exp_val_time(results, H_list, info, ket_list):
    """
    Calculate the expectation value of the Hamiltonian at various times.

    Parameters
    ----------
    H_list: list
        List of Hamiltonians.
    tlist: list
        List of times.
    ket_list: list
        List of kets at each time.

    Returns
    -------
    exp_val: list
        List of expectation values.
    """
    t_list=info[13]
    
    exp_val = []
    for t, ket in zip(tlist, ket_list):
        exp_val.append([H.expect(ket) for H in H_list])
    return exp_val


def exp_val_time_file(H_list, tlist, ket_list, file_name):
    """
    Calculate the expectation value of the Hamiltonian at various times and write to file.

    Parameters
    ----------
    H_list: list
        List of Hamiltonians.
    tlist: list
        List of times.
    ket_list: list
        List of kets at each time.
    file_name: str
        Name of file to write output to.
    """
    exp_val = exp_val_time(H_list, tlist, ket_list)
    with open(file_name, 'w') as f:
        for t, val in zip(tlist, exp_val):
            f.write(str(t) + ' ' + ' '.join([str(v) for v in val]) + '\n')
