# Heleen Oude Nijhuis, 12942936, Sanne Donker, 10780416, Introduction to computational science, Assignment 2

# Importing libraries and packages
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

#Event-driven simulation
#Parameters
time_steps = 10000
time = 0
N = 1000
Y = 50
Z = 1
X = N - Z - Y
beta = 0.75
gamma = 0.1
mu = 0.02
lamdba = 0.25

X_plot = []
Y_plot = []
Z_plot = []
time_steps2 = []

#Simulation
for i in range(time_steps):
    random_variable = np.random.uniform(0,1,7)
    if X <= 1:
        X = 0.00001
    if Y <= 1:
        Y = 0.00001
    if Z <= 1:
        Z = 0.00001
    delta_infection = -1/(beta*X*Y/N)*np.log(random_variable[1])
    delta_recovered = -1/(gamma*Y)*np.log(random_variable[2])
    delta_deathX = -1/(mu*X)*np.log(random_variable[3])
    delta_deathY = -1/(mu*Y)*np.log(random_variable[4])
    delta_deathZ = -1/(mu*Z)*np.log(random_variable[5])
    delta_birth = -1/(mu*N)*np.log(random_variable[6])
    var = {delta_infection:1,delta_recovered:2,delta_deathX:3,delta_deathY:4,delta_deathZ:5,delta_birth:6}
    u = var.get(min(var))
    time = time + min(var)
    time_steps2.append(time)

    if u == 1:
        X = X - 1
        Y = Y + 1
    if u == 2:
        Y = Y - 1
        Z = Z + 1
    if u == 3:
        X = X - 1
    if u == 4:
        Y = Y - 1
    if u == 5:
        Z = Z - 1
    if u == 6:
        X = X + 1
        
    X_plot.append(X)
    Y_plot.append(Y)
    Z_plot.append(Z)

# Plot the results
plt.plot(time_steps2, X_plot, 'y', label='Susceptible')
plt.plot(time_steps2, Y_plot, 'r', label='Infected')
plt.plot(time_steps2, Z_plot, 'g', label='Recovered')

    
plt.xlabel('Days')
plt.ylabel('Number of people')
plt.legend()
plt.grid()
plt.show()
