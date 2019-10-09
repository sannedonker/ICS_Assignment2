import framework
import numpy as np
import statistics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def variability(runs, initial_values, t, beta, gamma):
    """
    Plots the mean, max, and min value over runs number of runs of X
    """

    # make a list with runs list of X, Y, Z and N
    X_list = []
    Y_list = []
    Z_list = []
    N_list = []
    for i in range(runs):
        outcome = framework.sir_event_demo(initial_values, t, beta, gamma)

        X_list.append(outcome[0])
        Y_list.append(outcome[1])
        Z_list.append(outcome[2])
        N_list.append(outcome[3])

    # timesteps = []
    mean_X = []
    max_X = []
    min_X = []
    for i in range(len(t)):
        timestep = []
        for j in range(runs):
            timestep.append(X_list[j][i])
        mean_X.append(statistics.mean(timestep))
        max_X.append(max(timestep))
        min_X.append(min(timestep))

    plt.plot(t, mean_X, 'b', label="Mean")
    plt.plot(t, max_X, 'r', label="Max")
    plt.plot(t, min_X, 'g', label="Min")
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Population", fontsize=12)
    plt.legend()

    plt.show()


def variability2(runs, initial_values, t, beta, gamma):
    """
    Plots the mean, max, and min value over runs number of runs of X, Y, Z
    For constant population, if pop not constant uncomment the N_list lines
    """

    # make a list with runs list of X, Y, Z and N
    X_list = []
    Y_list = []
    Z_list = []
    N_list = []
    for i in range(runs):
        outcome = framework.sir_event_demo(initial_values, t, beta, gamma)

        X_list.append(outcome[0])
        Y_list.append(outcome[1])
        Z_list.append(outcome[2])
        N_list.append(outcome[3])

    outcomes = [X_list, Y_list, Z_list]

    # plot for every class
    for k in range(len(outcomes)):

        # save mean, max and min per timestep
        mean_list = []
        max_list = []
        min_list = []
        for i in range(len(t)):
            timestep = []
            for j in range(runs):
                timestep.append(outcomes[k][j][i])
            mean_list.append(statistics.mean(timestep))
            max_list.append(max(timestep))
            min_list.append(min(timestep))

        # plot mean, max and min
        mean_line = plt.plot(t, mean_list, 'b')
        max_line = plt.plot(t, max_list, 'r')
        min_line = plt.plot(t, min_list, 'g')

    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Population", fontsize=12)

    blue = mpatches.Patch(color="b", label="Mean")
    red = mpatches.Patch(color="r", label="Max")
    green = mpatches.Patch(color="g", label="Min")
    plt.legend(handles=[blue, red, green])

    plt.show()


N0 = 1000
Y0 = 100
Z0 = 0
X0 = N0 - Y0 - Z0
y0 = X0, Y0, Z0, N0

t = np.linspace(0, 1000, 10000)
beta = 3
gamma = 1
mu = 1 / 60
variability2(3, y0, t, beta, gamma)
