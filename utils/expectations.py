



# this file will contain funciton for returning the expectation values of the various Hamiltonians in time.


def exp_val_time(results, H_list, info):
    
    tlist = info[13]
    exp_val = []
    for i, _ in enumerate(tlist[:-1]):
        exp_val.append([H.expect(results.states[i]) for H in H_list])
    return exp_val

    t_list=info[13]
    
    exp_val = []
    for idx in range(len(tlist)):

        exp_val.append([H.expect(ket) for H in H_list])
    return exp_val


def exp_val_time_file(H_list, tlist, ket_list, file_name):
    

    exp_val = exp_val_time(H_list, tlist, ket_list)
    with open(file_name, 'w') as f:
        for t, val in zip(tlist, exp_val):
            f.write(str(t) + ' ' + ' '.join([str(v) for v in val]) + '\n')
