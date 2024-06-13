import matplotlib.pyplot as plt

def plot_hamiltonians(H_list):
    H_tot=H_list[1].full()
    H_tot = abs(H_tot)
    H_s=H_list[2].full()
    H_s = abs(H_s)
    H_int=H_list[3].full()
    H_int = abs(H_int)
    H_e=H_list[4].full()
    H_e = abs(H_e)
    H_s_self=H_list[5].full()
    H_s_self = abs(H_s_self)
    H_int_s=H_list[6].full()
    H_int_s = abs(H_int_s)
    H_int_e=H_list[7].full()
    H_int_e = abs(H_int_e)
    H_e_self=H_list[8].full()
    H_e_self = abs(H_e_self)
    #d, H_total, H_s, H_int, H_e, H_s_self, H_int_s, H_int_e, H_e_self
    # Assuming H_system_2 is the matrix you want to visualize
    plt.imshow(H_tot, cmap='hot', interpolation='nearest')
    plt.title('H_tot')
    plt.colorbar()
    plt.show()

    plt.imshow(H_s, cmap='hot', interpolation='nearest')
    plt.title('H_s')
    plt.colorbar()
    plt.show()

    plt.imshow(H_int, cmap='hot', interpolation='nearest')
    plt.title('H_int')
    plt.colorbar()
    plt.show()

    plt.imshow(H_e, cmap='hot', interpolation='nearest')
    plt.title('H_e')
    plt.colorbar()
    plt.show()

    plt.imshow(H_s_self, cmap='hot', interpolation='nearest')
    plt.title('H_s_self')
    plt.colorbar()
    plt.show()

    plt.imshow(H_int_s, cmap='hot', interpolation='nearest')
    plt.title('H_int_s')
    plt.colorbar()
