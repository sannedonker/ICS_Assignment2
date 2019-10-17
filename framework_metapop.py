# Importing libraries and packages
import scipy.integrate
import scipy.optimize

from scipy.integrate import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import pylab as plb
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.fftpack import fft
import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 300




def metapop(pop_y, t):

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

            # calculate labda
            sum = 0
            for j in range(len(pop_y)):
                sum += rho[i][j] * pop_y[j].get("Y")[-1] / N
            labda = beta * sum

            rate_list = [mu * N, labda * X , gamma * Y, mu * X, mu * Y, mu * Z]
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


# EXPERIMENT 1
N = 1000
Y = 100
Z = 0
X = N - Y - Z

t = 50
beta = 3
gamma = 1
mu = 0.05
rho = [[1, 0.1], [0.1, 1]]

pop_1 = {"X" : [X], "Y" : [Y], "Z" : [Z], "N" : [N]}
pop_2 = {"X" : [N], "Y" : [0], "Z" : [Z], "N" : [N]}

pop_y = [pop_1, pop_2]

pop_dict_list, time_list = metapop(pop_y, t)

zoom = int(len(time_list) / 3)
for i in range(len(pop_dict_list)):
    plt.plot(time_list[:zoom], pop_dict_list[i].get("Y")[:zoom], label="Population " + str(i + 1))

plt.legend()
plt.xlabel("Time", fontsize=12)
plt.ylabel("Number of infected individuals", fontsize=12)
plt.savefig("metapop_exp1_zoom.png", dpi=300)
plt.show()

for i in range(len(pop_dict_list)):
    plt.plot(time_list, pop_dict_list[i].get("Y"), label="Population " + str(i + 1))

plt.legend()
plt.xlabel("Time", fontsize=12)
plt.ylabel("Number of infected individuals", fontsize=12)
plt.savefig("metapop_exp1_no_zoom.png", dpi=300)
plt.show()

# # EXPERIMENT 2
# N = 1000
# Y = 100
# Z = 0
# X = N - Y - Z
#
# t = 50
# beta = 3
# gamma = 1
# mu = 0.05
# rho = [[1, 0.1, 0.9, 0, 0], [0.1, 1, 0, 0.1, 0], [0.9, 0, 1, 0, 0.9], [0, 0.1, 0, 1, 0], [0, 0, 0.9, 0, 1]]
#
# pop_1 = {"X" : [X], "Y" : [Y], "Z" : [Z], "N" : [N], "beta" : 3, "gamma" : 1, "mu" : 0.05}
# pop_2 = {"X" : [N], "Y" : [0], "Z" : [Z], "N" : [N], "beta" : 3, "gamma" : 1, "mu" : 0.05}
# pop_3 = {"X" : [N], "Y" : [0], "Z" : [Z], "N" : [N], "beta" : 3, "gamma" : 1, "mu" : 0.05}
# pop_4 = {"X" : [N], "Y" : [0], "Z" : [Z], "N" : [N], "beta" : 3, "gamma" : 1, "mu" : 0.05}
# pop_5 = {"X" : [N], "Y" : [0], "Z" : [Z], "N" : [N], "beta" : 3, "gamma" : 1, "mu" : 0.05}
#
# pop_y = [pop_1, pop_2, pop_3, pop_4, pop_5]
#
# pop_dict_list, time_list = metapop(pop_y, t)
#
# for i in range(len(pop_dict_list)):
#     plt.plot(time_list, pop_dict_list[i].get("Y"), label="Population " + str(i + 1))
#
# plt.legend()
# plt.xlabel("Time", fontsize=12)
# plt.ylabel("Number of infected individuals", fontsize=12)
# plt.savefig("metapop_t_50_2.png", dpi=300)
# plt.show()
