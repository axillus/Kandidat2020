import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd
import csv

from read_data import data
from model_version import model, model_info, guess_k_array


def read_results(constants):
    vald_modell, num_coefficient, num_tidserier, num_tidsteg, h = constants
    filnamn = "Optimering_viktad_model_" + vald_modell + ".csv"
    with open(filnamn) as csvfile:
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
    vald_modell, num_coefficient, num_tidserier, num_tidsteg, h = constants
    t_span, t_eval, y0 = ode_info
    sol = integrate.solve_ivp(fun=lambda t, y: model(vald_modell, t, y, kinetic_constants_0), t_span=t_span, y0=y0,
                              method="RK45",
                              t_eval=t_eval)
    sol_k = np.empty([num_tidserier, len(t_eval), 1])
    sol_k[:, :, 0] = sol.y
    return sol_k


def calc_residual(sol_k, constants, data_concentration, data_info):
    # beräknar residualen för de relevanta tidserierna
    vald_modell, num_coefficient, num_tidserier, num_tidsteg, h = constants
    compare_to_data, num_compare = data_info
    mat_r = np.zeros([num_compare, num_tidsteg, 1])
    compare = 0
    for tidsserie in range(num_tidserier):
        if compare_to_data[tidsserie] != False:
            data = data_concentration[int(compare_to_data[tidsserie]), :, 0]
            cut_data = data[~np.isnan(data)]
            num_data_tidsteg = len(cut_data)
            mat_r[compare, 0:num_data_tidsteg, 0] = cut_data[:] - sol_k[tidsserie, 0:num_data_tidsteg, 0]
            compare += 1
    return mat_r


def plot_data(data_concentration, time_points, figs, axes):
    fig_plot_all, fig_plot_compare, fig_plot_mig1, fig_plot_suc2 = figs
    ax_plot_all, ax_plot_compare, ax_plot_mig1, ax_plot_suc2 = axes
    data_mig1, data_hxk1, data_suc2 = data_concentration
    cb_palette = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
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


def plot_residual(mat_r, time_points, figs_residual, axes_reidual):
    fig_residual_mig1, fig_residual_suc2 = figs_residual
    ax_residual_mig1, ax_residual_suc2 = axes_reidual
    residual_suc2, residual_mig1 = mat_r
    # Mig1
    plt.figure(num=5)
    ax_residual_mig1.scatter(time_points, residual_mig1)
    # Suc2
    plt.figure(num=6)
    ax_residual_suc2.scatter(time_points, residual_suc2)


def plot_all(constants, sol_k, mat_r, data_concentration, time_points):
    vald_modell, num_coefficient, num_tidserier, num_tidsteg, h = constants
    save_directory = "Figurer_optimering/Modell_" + vald_modell + "/"
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
    plot_best_fit(sol_k, time_points, figs, axes)
    fig_residual_mig1 = plt.figure(num=5)
    ax_residual_mig1 = plt.axes()
    fig_residual_suc2 = plt.figure(num=6)
    ax_residual_suc2 = plt.axes()
    figs_residual = [fig_residual_mig1, fig_residual_suc2]
    axes_reidual = [ax_residual_mig1, ax_residual_suc2]
    plot_residual(mat_r, time_points, figs_residual, axes_reidual)

    ax_plot_all.legend(["Mig1", "Suc2", "Mig1p", "X"])
    ax_plot_all.set_xlabel("Tid (min)")
    ax_plot_all.set_ylabel("Intensitet")
    ax_plot_compare.legend(["Data Mig1", "Data Suc2", "Mig1", "Suc2"])
    ax_plot_compare.set_xlabel("Tid (min)")
    ax_plot_compare.set_ylabel("Intensitet")
    ax_plot_mig1.legend(["Data Mig1", "Mig1"])
    ax_plot_mig1.set_xlabel("Tid (min)")
    ax_plot_mig1.set_ylabel("Intensitet")
    ax_plot_suc2.legend(["Data Suc2", "Suc2"])
    ax_plot_suc2.set_xlabel("Tid (min)")
    ax_plot_suc2.set_ylabel("Intensitet")
    ax_residual_mig1.set_xlabel("Tid (min)")
    ax_residual_mig1.set_ylabel("Residual Mig1")
    ax_residual_suc2.set_xlabel("Tid (min)")
    ax_residual_suc2.set_ylabel("Residual Suc2")
    fig_plot_all.savefig(save_directory + "plot_all.eps", format='eps')
    ax_plot_all.set_ylim([-1, 20])
    fig_plot_all.savefig(save_directory + "plot_all_not_X.eps", format='eps')
    fig_plot_compare.savefig(save_directory + "compare.eps", format='eps')
    fig_plot_mig1.savefig(save_directory + "compare_mig1.eps", format='eps')
    fig_plot_suc2.savefig(save_directory + "compare_suc2.eps", format='eps')
    fig_residual_mig1.savefig(save_directory + "residual_mig1.eps", format='eps')
    fig_residual_suc2.savefig(save_directory + "residual_suc2.eps", format='eps')
    plt.show()


def get_min_cost(results):
    viktad_cost_funk = results[:, -1]
    index_min_viktad_cost_funk = viktad_cost_funk.argmin(axis=0)
    best_coefficients = results[index_min_viktad_cost_funk, 0:-2]
    min_cost_funk = results[index_min_viktad_cost_funk, -2]
    min_viktad_cost_funk = results[index_min_viktad_cost_funk, -1]
    print("kostnadsfunktionen är = " + str(min_cost_funk))
    print("Den viktade kostnadsfunktionen är = " + str(min_viktad_cost_funk))
    print("De bästa parametrarna är = " + ", ".join(repr(e) for e in best_coefficients))
    return best_coefficients, min_cost_funk, min_viktad_cost_funk


def main_plot_optimering():
    time_points, data_concentration = data()
    constants, ode_info, data_info = model_info(time_points)
    results = read_results(constants)
    coefficients, min_cost_funk, min_viktad_cost_funk = get_min_cost(results)
    sol_k = solve_ODE(coefficients, constants, ode_info)
    mat_r = calc_residual(sol_k, constants, data_concentration, data_info)
    plot_all(constants, sol_k, mat_r, data_concentration, time_points)


main_plot_optimering()

