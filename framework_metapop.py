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
    """
    """

    # # DIT IS ALLEEN VOOR TESTEN
    # birth = 0
    # transmission = 0
    # recovery = 0
    # deathX = 0
    # deathY = 0
    # deathZ = 0

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

            # ONDERSTAANDE ALLEEN ALS BETA GAMMA ENZO VERSCHILLEN
            # beta = pop.get("beta")
            # gamma = pop.get("gamma")
            # mu = pop.get("mu")

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
                # print(next_event)

                if next_event == 0:
                    X = X + 1
                    N = N + 1
                    # birth += 1

                elif next_event == 1:
                    X = X - 1
                    Y = Y + 1
                    # transmission += 1

                elif next_event == 2:
                    Y = Y - 1
                    Z = Z + 1
                    # recovery += 1

                elif next_event == 3:
                    X = X - 1
                    N = N -1
                    # deathX += 1

                elif next_event == 4:
                    Y = Y - 1
                    N = N - 1
                    # deathY += 1

                else:
                    Z = Z - 1
                    N = N - 1
                    # deathZ += 1

            pop.get("X").append(X)
            pop.get("Y").append(Y)
            pop.get("Z").append(Z)
            pop.get("N").append(N)

            i += 1

        time += min(dt)
        time_list.append(time)

    # print(birth, transmission, recovery, deathX, deathY, deathZ)

    return pop_y, time_list


# Parameters question 1
N = 1000
Y = 100
Z = 0
X = N - Y - Z

t = 50
beta = 3
gamma = 1
mu = 0.05
rho = [[1, 0.1], [0.1, 1]]

# ALS BETA, GAMMA EN MU ALIJTD HETZELFDE ZIJN KOST DIT VEEL EXTRA TIJD DUS ERUIT HALEN
pop_1 = {"X" : [X], "Y" : [Y], "Z" : [Z], "N" : [N], "beta" : 3, "gamma" : 1, "mu" : 0.05}
pop_2 = {"X" : [N], "Y" : [0], "Z" : [Z], "N" : [N], "beta" : 3, "gamma" : 1, "mu" : 0.05}


# Parameters question 2
N = 1000
Y = 100
Z = 0
X = N - Y - Z

t = 10
beta = 3
gamma = 1
mu = 0.05
rho = [[1, 0.4, 0.4, 0, 0.1], [0.4, 1, 0, 0.4, 0], [0.4, 0, 1, 0, 0.1], [0, 0.4, 0, 1, 0], [0.4, 0, 0.1, 0, 1]]

# ALS BETA, GAMMA EN MU ALIJTD HETZELFDE ZIJN KOST DIT VEEL EXTRA TIJD DUS ERUIT HALEN
pop_1 = {"X" : [X], "Y" : [Y], "Z" : [Z], "N" : [N], "beta" : 3, "gamma" : 1, "mu" : 0.05}
pop_2 = {"X" : [N], "Y" : [0], "Z" : [Z], "N" : [N], "beta" : 3, "gamma" : 1, "mu" : 0.05}
pop_3 = {"X" : [N], "Y" : [0], "Z" : [Z], "N" : [N], "beta" : 3, "gamma" : 1, "mu" : 0.05}
pop_4 = {"X" : [N], "Y" : [0], "Z" : [Z], "N" : [N], "beta" : 3, "gamma" : 1, "mu" : 0.05}
pop_5 = {"X" : [N], "Y" : [0], "Z" : [Z], "N" : [N], "beta" : 3, "gamma" : 1, "mu" : 0.05}


pop_y = [pop_1, pop_2, pop_3, pop_4, pop_5]

pop_dict_list, time_list = metapop(pop_y, t)

for i in range(len(pop_dict_list)):
    plt.plot(time_list, pop_dict_list[i].get("Y"), label="Population " + str(i + 1))

plt.legend()
plt.show()
