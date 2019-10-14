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

    time = 0
    time_list = [0]
    # counter = 0
    while time < t:

        # birth, transmission, recovery, death X, death Y, death Z
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
            N = N + 1

        elif first_event == 1:
            X = X - 1
            Y = Y + 1

        elif first_event == 2:
            Y = Y - 1
            Z = Z + 1

        elif first_event == 3:
            X = X - 1
            N = N -1

        elif first_event == 4:
            Y = Y - 1
            N = N - 1

        else:
            Z = Z - 1
            N = N - 1

        X_list.append(X)
        Y_list.append(Y)
        Z_list.append(Z)
        N_list.append(N)

        time += min(dt)
        time_list.append(time)

    return X_list, Y_list, Z_list, N_list, time_list


# Discrete event simulation with infection induced death
def sir_event_demo_die(y, t, beta, gamma):

    X, Y, Z, N = y
    X_list = [X]
    Y_list = [Y]
    Z_list = [Z]
    N_list = [N]

    time = 0
    time_list = [0]
    counter = 0
    while time < t:

        # birth, transmission, recovery, death X, death Y, death Z, death while infected
        rates = [mu * N, beta * X * Y / N, gamma * Y, mu * X, mu * Y, mu * Z, Y / (1 - rho)]

        dt = []
        for j in range(len(rates)):
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

        time += min(dt)
        time_list.append(time)

    return X_list, Y_list, Z_list, N_list, time_list

# Discrete event simulation
def sir_event_demo_die_imports(y, t, beta, gamma):

    X, Y, Z, N = y
    X_list = [X]
    Y_list = [Y]
    Z_list = [Z]
    N_list = [N]

    time = 0
    time_list = [0]
    counter = 0
    birth = 0
    transmission = 0
    recovery = 0
    deathX = 0
    deathY = 0
    deathZ = 0
    deathInfected = 0
    imports = 0
    passingthrough = 0
    while time < t:

        # birth, transmission, recovery, death X, death Y, death Z, death while infected
        # import, passing through
        rates = [mu * N, beta * X * Y / N, gamma * Y, mu * X, mu * Y, mu * Z, Y / (1 - rho),  delta, epsilon * X]

        dt = []
        for j in range(len(rates)):
            u = np.random.uniform(0, 1)
            if rates[j] < 0.0001:
                dt.append(100000)
                counter += 1
            else:
                dt.append(-np.log(u) / rates[j])

        first_event = dt.index(min(dt))
        # print(first_event)

        if first_event == 0:
            X = X + 1
            N = N + 1
            birth += 1

        elif first_event == 1:
            X = X - 1
            Y = Y + 1
            transmission += 1

        elif first_event == 2:
            Y = Y - 1
            Z = Z + 1
            recovery += 1

        elif first_event == 3:
            X = X - 1
            N = N - 1
            deathX += 1

        elif first_event == 4:
            Y = Y - 1
            N = N - 1
            deathY += 1

        elif first_event == 5:
            Z = Z - 1
            N = N - 1
            deathZ += 1

        elif first_event == 6:
            Y = Y - 1
            N = N + 1
            deathInfected += 1

        elif first_event == 7:
            Y = Y + 1
            N = N + 1
            imports += 1

        else:
            X = X - 1
            Y = Y + 1
            passingthrough += 1

        X_list.append(X)
        Y_list.append(Y)
        Z_list.append(Z)
        N_list.append(N)

        time += min(dt)
        time_list.append(time)

    print(birth, transmission, recovery, deathX, deathY, deathZ, deathInfected, imports, passingthrough)
    return X_list, Y_list, Z_list, N_list, time_list


# Discrete event simulation
def sir_event_demo_imports(y, t, beta, gamma):

    X, Y, Z, N = y
    X_list = [X]
    Y_list = [Y]
    Z_list = [Z]
    N_list = [N]

    time = 0
    time_list = [0]
    counter = 0
    birth = 0
    transmission = 0
    recovery = 0
    deathX = 0
    deathY = 0
    deathZ = 0
    deathInfected = 0
    imports = 0
    passingthrough = 0
    while time < t:

        # birth, transmission, recovery, death X, death Y, death Z, death while infected
        # import, passing through
        rates = [mu * N, beta * X * Y / N, gamma * Y, mu * X, mu * Y, mu * Z, delta, epsilon * X]

        dt = []
        for j in range(len(rates)):
            u = np.random.uniform(0, 1)
            if rates[j] < 0.0001:
                dt.append(100000)
                counter += 1
            else:
                dt.append(-np.log(u) / rates[j])

        first_event = dt.index(min(dt))
        # print(first_event)

        if first_event == 0:
            X = X + 1
            N = N + 1
            birth += 1

        elif first_event == 1:
            X = X - 1
            Y = Y + 1
            transmission += 1

        elif first_event == 2:
            Y = Y - 1
            Z = Z + 1
            recovery += 1

        elif first_event == 3:
            X = X - 1
            N = N - 1
            deathX += 1

        elif first_event == 4:
            Y = Y - 1
            N = N - 1
            deathY += 1

        elif first_event == 5:
            Z = Z - 1
            N = N - 1
            deathZ += 1

        elif first_event == 6:
            Y = Y + 1
            N = N + 1
            imports += 1

        else:
            X = X - 1
            Y = Y + 1
            passingthrough += 1

        X_list.append(X)
        Y_list.append(Y)
        Z_list.append(Z)
        N_list.append(N)

        time += min(dt)
        time_list.append(time)

    print(birth, transmission, recovery, deathX, deathY, deathZ, imports, passingthrough)
    return X_list, Y_list, Z_list, N_list, time_list


N0 = 10000
Y0 = 100
Z0 = 0
X0 = N0 - Y0 - Z0
y0 = X0, Y0, Z0, N0


t = 100
# beta = 1 / 3
# gamma = 0.01
beta = 1/2
gamma = 1 / 6
mu = 5e-2
# rho = 0.3
delta = 0.01
epsilon = 0.001

def diff(y, t, beta, gamma):
    S, I, R, N = y
    dSdt = mu * N -beta * S * I / N - mu * S
    dIdt = beta * S * I / N - gamma * I - mu * I
    dRdt = gamma * I - mu * R
    dNdt = dSdt + dIdt + dRdt
    return [dSdt, dIdt, dRdt, dNdt]

t = 150
X, Y, Z, N, time_list = sir_event_demo(y0, t, beta, gamma)
# print(time_list)

plt.plot(X, Y, 'r', label="Stochastic")

t = np.linspace(0, 150, 1000)

ret = odeint(diff, y0, t, args=(beta, gamma))
X, Y, Z, N = ret.T
plt.plot(X, Y, 'b', label="Deterministic")


# # NEGATIVE COVARIANCE EXPERIMENT
# def negative_covariance(X, Y):
#     two_d = np.vstack((np.asarray(X), np.asarray(Y)))
#     print(np.cov(two_d))
#
# negative_covariance(X, Y)

# PHASEPOLOT INCREASED TRANSIENTS
# plt.plot(X, Y)

plt.xlabel("Number of susceptible individuals", fontsize=12)
plt.ylabel("Number of infected individuals", fontsize=12)
plt.ticklabel_format(style='sci', scilimits=(0,0))
plt.legend()
plt.savefig("phaseplot_transients.png", dpi=300)
plt.show()



#  # plot S(t), I(t) and R(t)
# plt.plot(time_list, X, 'b', label="Susceptible")
# plt.plot(time_list, Y, 'r', label="Infected")
# # plt.plot(time_list, Z, 'g', label="Recovered")
# # plt.plot(time_list, N, 'y', label="Population")
# plt.xlabel("Time", fontsize=12)
# plt.ylabel("Population", fontsize=12)
# plt.grid(True, axis='x')
# plt.legend()
#
# plt.savefig("negative_covariance.png", dpi=300)
# plt.show()
