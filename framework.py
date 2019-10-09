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



# Discrete event simulation
def sir_event_demo(y, t, beta, gamma):

    X, Y, Z, N = y
    X_list = [X]
    Y_list = [Y]
    Z_list = [Z]
    N_list = [N]

    # birth, transmission, recovery, death X, death Y, death Z
    rates = [mu * N, beta * X * Y / N, gamma * Y, mu * X, mu * Y, mu * Z]

    time = 0
    time_list = [0]
    # counter = 0
    while time < t:

        rates = [mu * N, beta * X * Y / N, gamma * Y, mu * X, mu * Y, mu * Z]

        dt = []
        for j in range(6):
            u = np.random.uniform(0, 1)
            if rates[j] < 0.0001:
                dt.append(100000)
                # counter += 1
            else:
                dt.append(-np.log(u) / rates[j])

        first_event = dt.index(min(dt))

        if first_event == 0:
            X = X + 1

        elif first_event == 1:
            X = X - 1
            Y = Y + 1

        elif first_event == 2:
            Y = Y - 1
            Z = Z + 1

        elif first_event == 3:
            X = X - 1

        elif first_event == 4:
            Y = Y - 1

        else:
            Z = Z - 1

        X_list.append(X)
        Y_list.append(Y)
        Z_list.append(Z)
        N_list.append(N)

        time += min(dt)
        time_list.append(time)

    return X_list, Y_list, Z_list, N_list, time_list


# NOG DE OUDE TIJDSCHAAL --> DUS FOUT!!!!
# Discrete event simulation
def sir_event_demo_die(y, t, beta, gamma):

    X, Y, Z, N = y
    X_list = []
    Y_list = []
    Z_list = []
    N_list = []

    # birth, transmission, recovery, death X, death Y, death Z, death while infected
    rates = [mu * N, beta * X * Y / N, gamma * Y, mu * X, mu * Y, mu * Z, Y / (rho - 1)]

    counter = 0
    for i in t:

        rates = [mu * N, beta * X * Y / N, gamma * Y, mu * X, mu * Y, mu * Z]

        dt = []
        for j in range(6):
            u = np.random.uniform(0, 1)
            if rates[j] < 0.0001:
                dt.append(100000)
                counter += 1
            else:
                dt.append(-np.log(u) / rates[j])

        first_event = dt.index(min(dt))

        if first_event == 0:
            X = X + 1
            N = N + 1

        elif first_event == 1:
            X = X - 1
            Y = Y + 1

        elif first_event == 2:
            Y = Y - 1
            Z = Z + 1

        elif first_event == 3:
            X = X - 1
            N = N - 1

        elif first_event == 4:
            Y = Y - 1
            N = N - 1

        elif first_event == 5:
            Z = Z - 1
            N = N - 1

        else:
            Y = Y - 1
            N = N + 1

        X_list.append(X)
        Y_list.append(Y)
        Z_list.append(Z)
        N_list.append(N)

        t += min(dt)

    return X_list, Y_list, Z_list, N_list



N0 = 1000
Y0 = 100
Z0 = 0
X0 = N0 - Y0 - Z0
y0 = X0, Y0, Z0, N0

# t = np.linspace(0, 1000, 10000)
t = 1000
beta = 5
gamma = 1 / 2
mu = 1 / 50
rho = 0.1


X, Y, Z, N, time_list = sir_event_demo(y0, t, beta, gamma)
print(time_list)

 # plot S(t), I(t) and R(t)
plt.plot(time_list, X, 'b', label="Susceptible")
plt.plot(time_list, Y, 'r', label="Infected")
plt.plot(time_list, Z, 'g', label="Recovered")
plt.plot(time_list, N, 'y', label="Population")
plt.xlabel("Time", fontsize=12)
plt.ylabel("Population", fontsize=12)
plt.legend()

plt.show()
