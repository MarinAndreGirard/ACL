import matplotlib.pyplot as plt

#TODO
#-Add more complex ploting functions

def plot_basic(x, y, x_label, y_label, title, fig_size=(10, 5)):
    """
    Plots the given parameters.
    
    Args:
        x (array): x-axis values.
        y (array): y-axis values.
        x_label (str): label for the x-axis.
        y_label (str): label for the y-axis.
        title (str): title for the plot.
    """
    plt.figure(figsize=fig_size)
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

#Make a plot function that also takes a bninning number and bins close values together.

def plot_with_binning(x,x_label, title, fig_size=(10, 5), bin_number=100):
    """
    Plots the given parameters after binning the y values.
    
    Args:
        x (array): x-axis values.
        y (array): y-axis values.
        x_label (str): label for the x-axis.
        y_label (str): label for the y-axis.
        title (str): title for the plot.
        bin_number (int): number of bins to bin the y values.
    """
    plt.figure(figsize=fig_size)
    plt.hist(x, bins=bin_number)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Count")
    plt.show()
