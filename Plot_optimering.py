import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd
import csv

from read_data import full_data
from model_version import model, model_info, guess_k_array


def read_results():
    with open("viktad_model_1.csv") as csvfile:
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
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = integrate.solve_ivp(fun=lambda t, y: model(t, y, kinetic_constants_0), t_span=t_span, y0=y0,
                                method="RK45",
                                t_eval=t_eval)
    sol_k = np.empty([num_tidserier, len(t_eval), 1])
    sol_k[:, :, 0] = sol.y
    return sol_k, t_eval


def plot_data(data_concentration, time_points, figs, axes):
    fig_plot_all, fig_plot_compare, fig_plot_mig1, fig_plot_suc2 = figs
    ax_plot_all, ax_plot_compare, ax_plot_mig1, ax_plot_suc2 = axes
    data_mig1, data_hxk1, data_suc2 = data_concentration
    cb_palette = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    # All
    plt.figure(num=1)
    ax_plot_all.plot(time_points, data_mig1, color=cb_palette[0], linestyle="--", marker="D")
    ax_plot_all.plot(time_points, data_hxk1, color=cb_palette[1], linestyle="--", marker="D")
    ax_plot_all.plot(time_points, data_suc2, color=cb_palette[2], linestyle="--", marker="D")
    # Compare
    plt.figure(num=2)
    ax_plot_compare.plot(time_points, data_mig1, color=cb_palette[0], linestyle="--", marker="D")
    ax_plot_compare.plot(time_points, data_suc2, color=cb_palette[2], linestyle="--", marker="D")
    # Mig1
    plt.figure(num=3)
    ax_plot_mig1.plot(time_points, data_mig1, color=cb_palette[0], linestyle="--", marker="D")
    # Suc2
    plt.figure(num=4)
    ax_plot_suc2.plot(time_points, data_suc2, color=cb_palette[2], linestyle="--", marker="D")


def plot_best_fit(sol_k, time_points, figs, axes):
    fig_plot_all, fig_plot_compare, fig_plot_mig1, fig_plot_suc2 = figs
    ax_plot_all, ax_plot_compare, ax_plot_mig1, ax_plot_suc2 = axes
    suc2, mig1, mig1_phos, X = sol_k
    cb_palette = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    # All
    plt.figure(num=1)
    ax_plot_all.plot(time_points, mig1, color=cb_palette[0])
    ax_plot_all.plot(time_points, suc2, color=cb_palette[2])
    ax_plot_all.plot(time_points, mig1_phos, color=cb_palette[3], alpha=0.5)
    ax_plot_all.plot(time_points, X, color=cb_palette[4], alpha=0.5)
    # Compare
    plt.figure(num=2)
    ax_plot_compare.plot(time_points, mig1, color=cb_palette[0])
    ax_plot_compare.plot(time_points, suc2, color=cb_palette[2])
    # Mig1
    plt.figure(num=3)
    ax_plot_mig1.plot(time_points, mig1, color=cb_palette[0])
    # Suc2
    plt.figure(num=4)
    ax_plot_suc2.plot(time_points, suc2, color=cb_palette[2])


def plot_all(sol_k, t_eval, data_concentration, time_points):
    fig_plot_all = plt.figure(num=1)
    ax_plot_all = plt.axes()
    fig_plot_compare = plt.figure(num=2)
    ax_plot_compare = plt.axes()
    fig_plot_mig1 = plt.figure(num=3)
    ax_plot_mig1 = plt.axes()
    fig_plot_suc2 = plt.figure(num=4)
    ax_plot_suc2 = plt.axes()
    figs = [fig_plot_all, fig_plot_compare, fig_plot_mig1, fig_plot_suc2]
    axes = [ax_plot_all, ax_plot_compare, ax_plot_mig1, ax_plot_suc2]
    plot_data(data_concentration, time_points, figs, axes)
    plot_best_fit(sol_k, t_eval, figs, axes)
    ax_plot_all.legend(["Data Mig1", "Data Hxk1", "Data Suc2", "Mig1", "Suc2", "Mig1P", "X"])
    ax_plot_compare.legend(["Data Mig1", "Data Suc2", "Mig1", "Suc2"])
    ax_plot_mig1.legend(["Data Mig1", "Mig1"])
    ax_plot_suc2.legend(["Data Suc2", "Suc2"])
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
    sol_k, t_eval = solve_ODE(coefficients, constants, ode_info)
    plot_all(sol_k, t_eval, data_concentration, time_points)




main_plot_optimering()