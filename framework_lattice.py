import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def get_neighbor_matrix(n, rho):
    """
    Get matrix with index rho for (i, j) and (j, i) if i and j are neigbors
    """

    # generate lattice of n x n and it's corresponding dict of dicts
    lattice = nx.generators.lattice.grid_graph([n, n])
    adjacency_dict = nx.convert.to_dict_of_dicts(lattice)

    # make adjacency matrix of lattice
    neighbor_matrix = []
    for i in range(n):
        for j in range(n):
            values = adjacency_dict.get((i, j))
            row_neighbor_matrix = []
            for k in range(n):
                for l in range(n):
                    if values.get((k, l)) == {}:
                        row_neighbor_matrix.append(rho)
                    else:
                        row_neighbor_matrix.append(0)
            neighbor_matrix.append(row_neighbor_matrix)

    return neighbor_matrix

def get_rho_matrix(n, rho):
    """
    Add ones to diagonal
    """
    neighbors = get_neighbor_matrix(n, rho)
    for i in range(n * n):
        for j in range(n * n):
            if i == j:
                neighbors[i][j] = 1

    return neighbors

def generate_populations(n, start_infection, X, Y, Z, N):
    """
    Generate a metapopulation with n subpopulations
    start_infection is the subpopulation in which the infection starts

    ---------
    IF NEEDED ALSO BETA, GAMMA AND MU CAN BE GIVEN TO POPS
    IF NEEDED X, Y, Z, N CAN BE LISTS SO POPSIZES AREN'T THE SAME
    """

    pop_list = []
    for i in range(n):
        if i == start_infection:
            pop = {"X" : [X], "Y" : [Y], "Z" : [Z], "N" : [N]}
        else:
            pop = {"X" : [N], "Y" : [0], "Z" : [Z], "N" : [N]}
        pop_list.append(pop)

    return pop_list

def metapop(pop_y, rho, beta, gamma, mu, t):

    rho_matrix = get_rho_matrix(int(math.sqrt(len(pop_y))), rho)

    time = 0
    time_list = [0]
    while time < t:

        rates = []
        i = 0
        for pop in pop_y:
            X = pop.get("X")[-1]
            Y = pop.get("Y")[-1]
            Z = pop.get("Z")[-1]
            N = pop.get("N")[-1]

            # calculate force of infection
            sum_ji = 0
            sum_ij = 0
            for j in range(len(pop_y)):
                sum_ji += rho_matrix[j][i]
                sum_ij += rho_matrix[i][j] * pop_y[j].get("Y")[-1]
            foi = beta * X * ((1 - sum_ji) * Y + sum_ij) / N

            rate_list = [mu * N, foi , gamma * Y, mu * X, mu * Y, mu * Z]
            for rate in rate_list:
                rates.append(rate)

            i += 1

        dt = []
        for i in range(len(rates)):
            u = np.random.uniform(0, 1)
            if rates[i] < 0.0001:
                dt.append(100000)
            else:
                dt.append(-np.log(u) / rates[i])

        next_event = dt.index(min(dt))
        pop_event = next_event // len(rate_list)

        i = 0
        for pop in pop_y:

            # get X, Y, Z, N
            X = pop.get("X")[-1]
            Y = pop.get("Y")[-1]
            Z = pop.get("Z")[-1]
            N = pop.get("N")[-1]

            # the population in which the event happens
            if i == pop_event:
                next_event = next_event % len(rate_list)

                if next_event == 0:
                    X = X + 1
                    N = N + 1

                elif next_event == 1:
                    X = X - 1
                    Y = Y + 1

                elif next_event == 2:
                    Y = Y - 1
                    Z = Z + 1

                elif next_event == 3:
                    X = X - 1
                    N = N -1

                elif next_event == 4:
                    Y = Y - 1
                    N = N - 1

                else:
                    Z = Z - 1
                    N = N - 1

            pop.get("X").append(X)
            pop.get("Y").append(Y)
            pop.get("Z").append(Z)
            pop.get("N").append(N)

            i += 1

        time += min(dt)
        time_list.append(time)

    return pop_y, time_list

def lattice_model(n, rho, t, start_infection, X, Y, Z, N, beta, gamma, mu):

    pops = generate_populations(n * n, start_infection, X, Y, Z, N)
    pops_over_time, time_list = metapop(pops, rho, beta, gamma, mu, t)

    # comment this when function is used for calculating speed of infection
    # generate plot with the course of infection
    for i in range(len(pops_over_time)):
        plt.plot(time_list, pops_over_time[i].get("Y"), label="pop " + str(i))

    plt.legend(bbox_to_anchor=(1.05, 1), ncol = 2, fontsize=15)
    plt.savefig("lattice_sizes.png", dpi=300)
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Number of infected individuals", fontsize=18)
    plt.show()

    # comment this when function is used for calculating speed of infection
    # generate heatmap
    time_steps = int(len(time_list) / 10)
    heat_plots = []
    for i in range(10):
        print(i * time_steps)
        heatmap(pops_over_time, int(i * time_steps), "_sizes_" + str(i))

    # calculate speed of infection
    furthest_pop = pops_over_time[-1].get("Y")
    for i in range(len(furthest_pop)):
        if furthest_pop[i] != 0:
            return time_list[i]


def heatmap(pops, t, name):
    n = int(math.sqrt(len(pops)))
    matrix = []
    for i in range(len(pops)):
        if i % n == 0:
            if i != 0:
                matrix.append(row)
            row = [pops[i].get("Y")[t]]
        else:
            row.append(pops[i].get("Y")[t])

    matrix.append(row)

    matrix = np.array(matrix)
    plt.imshow(matrix)
    plt.colorbar()
    plt.draw()
    plt.clim(0, 50)
    plt.savefig("heatmap_" + name + ".png", dpi=300)
    plt.show()


N = 100
Y = 10
Z = 0
X = N - Y - Z

beta = 3
gamma = 1
mu = 0.05

t = 25
rho = 0.1
grid_size = 6
start_infection = 0
rho_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def speed_of_infection(n, rho_list):
    """
    Calculate the average of the speed of infection over n runs
    """
    averages = []
    for j in rho_list:
        distance_time = []
        for i in range(n):
            distance_time.append(lattice_model(grid_size, j, t, start_infection, X, Y, Z, N, beta, gamma, mu))
        average = sum(distance_time) / n
        averages.append(average)
    plt.plot(rho_list, averages)
    plt.xlabel("Rho", fontsize=12)
    plt.ylabel("Time (days)", fontsize=12)
    plt.show()
