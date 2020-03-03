from ODEs_0 import *
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


def solve_ODEs(conncentrations_t0, timespan, model_ode, kinetic_parameter):
    # Function that solves the ODEs in ODE file using initial values as "concentrations_t0[n]"
    # and plots the solutions.
    sol = solve_ivp(lambda t, concentration: model_ode(t, concentration, kinetic_parameter), timespan, conncentrations_t0, dense_output=True)  # Solver utilising RK45
    t = np.linspace(0, 10, 1000)  # Create timepoints to plot on
    conc = sol.sol(t)    # Transfer solution to a plottable variable
    plt.plot(t, conc.T)  # Plot species vs time
    plt.xlabel('t')      # X-axis title
    plt.legend(['Mig1', 'Mig1*', 'Suc2', 'X'])
    plt.title('Mig1 signalling pathway model')
    plt.show()


def main_solve():
    concentrations_t0 = [1, 0, 1, 0]  # Initial concentrations: Mig1, Mig1*, SUC2, X
    timespan= [0, 10]  # Span to solve ODEs on
    kinetic_parameter0=[0.1, 2.6, 1, 0.4, 1, 0.2, 1]
    solve_ODEs(concentrations_t0, timespan, pathway_ode0, kinetic_parameter0)


main_solve()