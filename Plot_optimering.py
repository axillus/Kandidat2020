import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd
import csv

from read_data import full_data
from model_version import model, model_info, guess_k_array


def read_results():
    with open("model_1.csv") as csvfile:
        result_reader = csv.reader(csvfile)
        initial = 0
        for row in result_reader:
            new_result = ",".join(row)
            if new_result != "":
                if initial == 0:
                    results = np.array([(",".join(row)).split(sep=",")]).astype(np.float64)
                    initial = 1
                else:
                    new_result_array = np.array(new_result.split(sep=",")).astype(np.float64)
                    results = np.vstack((results, new_result_array))
    return results


def solve_ODE(kinetic_constants_0, constants, ode_info):
    num_coefficient, num_tidserier, num_tidsteg, h = constants
    t_span, t_eval, y0 = ode_info
    sol = integrate.solve_ivp(fun=lambda t, y: model(t, y, kinetic_constants_0), t_span=t_span, y0=y0,
                                method="RK45",
                                t_eval=t_eval)
    sol_k = np.empty([num_tidserier, num_tidsteg, 1])
    sol_k[:, :, 0] = sol.y
    return sol_k


def plot_data(data_concentration, time_points, fig_and_axes):
    data_mig1, data_hxk1, data_suc2 = data_concentration
    print(data_mig1)
    cut_data_mig1 = data_mig1[~np.isnan(data_mig1)]
    print(cut_data_mig1)
    print(len(data_mig1))
    print(len(cut_data_mig1))
    print(data_mig1.shape)
    print(cut_data_mig1.shape)
    fig_plot, ax_plot = fig_and_axes
    plt.figure(num=1)
    ax_plot.plot(time_points, data_mig1)
    ax_plot.plot(time_points, data_hxk1)
    ax_plot.plot(time_points, data_suc2)


def plot_best_fit(sol_k, time_points, fig_and_axes):
    suc2, mig1, mig1_phos, X = sol_k
    fig_plot, ax_plot = fig_and_axes
    plt.figure(num=1)
    ax_plot.plot(time_points, suc2)
    ax_plot.plot(time_points, mig1)
    ax_plot.plot(time_points, mig1_phos)
    ax_plot.plot(time_points, X)


def plot_all(sol_k, data_concentration, time_points):
    fig_plot = plt.figure(num=1)
    ax_plot = plt.axes()
    fig_and_axes = [fig_plot, ax_plot]
    plot_data(data_concentration, time_points, fig_and_axes)
    plot_best_fit(sol_k, time_points, fig_and_axes)
    plt.show()


def get_min_cost(results):
    cost_funk = results[:, -1]
    index_min_cost = cost_funk.argmin(axis=0)
    best_coefficients = results[index_min_cost, 0:-1]
    min_cost = cost_funk[index_min_cost]
    return best_coefficients, min_cost


def main_plot_optimering():
    time_points, data_concentration = full_data()
    constants, ode_info, data_info = model_info(time_points)
    results = read_results()
    coefficients, min_cost = get_min_cost(results)
    sol_k = solve_ODE(coefficients, constants, ode_info)
    plot_all(sol_k, data_concentration, time_points)




main_plot_optimering()