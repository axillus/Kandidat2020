import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
import time

# välj start k-värden [k_1, k_2, k_3, ... k_i] i = antal parametrar
# kör ODE med givna värden
# välj steglängd h
# kör ODE i gånger med ett k_j+h ändrat i varje fall
# använd resultatet för att beräkna gradienten
# beräkna hessianen
# se om hessianen är lösbar, om inte...fan
# ta fram p = - H^-1*grad_f
# k_ny = k+p , om abs(k - k_ny) < limit : break
# starta från början

# få in constraind optimisation
# Lagrange multiplyer
# Lagrange dualproblem

# interior penalty methods, sid 330
# gradient projection method


def model(t, y, kinetic_constants):
    substance1 = y[0]
    substance2 = y[1]
    r1 = substance1*kinetic_constants[0]
    r2 = substance2*kinetic_constants[1]
    dsubstance1_dt = -r1 + r2
    dsubstance2_dt = r1 - r2
    dy_dt = [dsubstance1_dt, dsubstance2_dt]
    return dy_dt


def num_coeff():
    num_coefficient = 2
    return num_coefficient


def guess_k_array(gissning):
    if gissning == 0:
        k_array = np.array([50, 1], np.float64)
    elif gissning == 1:
        k_array = np.array([1, 50], np.float64)
    else:
        k_array = np.array([10, 10], np.float64)
    return k_array


def model_info(time_points):
    num_coefficient = num_coeff()
    num_tidserier = 2
    t_eval = time_points
    num_tidsteg = len(t_eval)
    eps = np.finfo(float).eps
    h = np.sqrt(eps)
    t_span = [t_eval[0], t_eval[-1]]
    substance1_0 = 4.043530
    substance2_0 = 2.649860
    y0 = np.array([substance1_0, substance2_0])
    compare_to_data = ["0", "1"]
    num_compare = 2
    constants = (num_coefficient, num_tidserier, num_tidsteg, h)
    ode_info = (t_span, t_eval, y0)
    data_info = (compare_to_data, num_compare)
    return constants, ode_info, data_info


def set_true_values():
    true_values = [10.454, 17.837]
    return true_values


def data():
    kinetic_constants_true = set_true_values()
    t_span = [0, 1]
    y0 = [4.043530, 2.649860]
    t_eval = np.linspace(0, 1, num=100)
    sol = integrate.solve_ivp(fun=lambda t, y: model(t, y, kinetic_constants_true), t_span=t_span, y0=y0, method="RK45",
                              t_eval=t_eval)
    data_conc = np.empty([2, 100, 1])
    # brus = np.random.normal(scale=0.01, size=(2, 100))
    data_conc[:, :, 0] = sol.y # + brus
    time_points = t_eval
    return time_points, data_conc


def calc_sol_k(kinetic_constants_0, constants, ode_info):
    num_coefficient, num_tidserier, num_tidsteg, h = constants
    t_span, t_eval, y0 = ode_info
    sol = integrate.solve_ivp(fun=lambda t, y: model(t, y, kinetic_constants_0), t_span=t_span, y0=y0, method="RK45",
                              t_eval=t_eval)
    sol_k = np.empty([num_tidserier, num_tidsteg, 1])
    krashade = False
    try:
        sol_k[:, :, 0] = sol.y
    except ValueError:
        krashade = True
    return sol_k, krashade


def calc_sol_k_step(kinetic_constants_0, constants, ode_info):
    num_coefficient, num_tidserier, num_tidsteg, h = constants
    t_span, t_eval, y0 = ode_info
    sol_k_step = np.empty([num_tidserier, num_tidsteg, num_coefficient])
    for k in range(num_coefficient):
        kinetic_constants_step = kinetic_constants_0.copy()
        kinetic_constants_step[k] = kinetic_constants_step[k] + h
        sol = integrate.solve_ivp(fun=lambda t, y: model(t, y, kinetic_constants_step), t_span=t_span, y0=y0,
                                  method="RK45", t_eval=t_eval)
        sol_k_step[:, :, k] = sol.y
    return sol_k_step


def calc_variance(constants, data_info):
    num_coefficient, num_tidserier, num_tidsteg, h = constants
    compare_to_data, num_compare = data_info
    mat_w = np.empty([num_compare, num_tidsteg, num_tidsteg])
    for compare in range(num_compare):
        mat_w[compare, :, :] = np.identity(num_tidsteg)
    return mat_w


def calc_gradient(sol_k, sol_k_step, mat_w, constants, data_concentration, data_info):
    # grad = - 2 * A_T * W * r
    num_coefficient, num_tidserier, num_tidsteg, h = constants
    compare_to_data, num_compare = data_info
    mat_r = calc_residual(sol_k, constants, data_concentration, data_info)
    diff_k_t = np.empty([num_compare, num_tidsteg, num_coefficient])
    mat_a = np.empty([num_compare, num_tidsteg, num_coefficient])  # mat_A = S_T, S = sensitivity matrix
    for k in range(num_coefficient):
        compare = 0
        for tidserie in range(num_tidserier):
            if compare_to_data[tidserie] != False:
                diff_k_t[compare, :, k] = sol_k_step[tidserie, :, k] - sol_k[tidserie, :, 0]
                compare += 1
        mat_a[:, :, k] = diff_k_t[:, :, k]/h
    mat_a_transpose = np.transpose(mat_a, (0, 2, 1))  # mat_A_T = S, S = sensitivity matrix
    mat_temp = np.matmul(mat_a_transpose, mat_w, axes=[(-2, -1), (-2, -1), (-2, -1)])
    mat_temp2 = np.matmul(mat_temp, mat_r, axes=[(-2, -1), (-2, -1), (-2, -1)])
    grad = -2 * np.add.reduce(mat_temp2, 0)
    return grad, mat_a, mat_a_transpose


def calc_residual(sol_k, constants, data_concentration, data_info):
    num_coefficient, num_tidserier, num_tidsteg, h = constants
    compare_to_data, num_compare = data_info
    mat_r = np.empty([num_compare, num_tidsteg, 1])
    compare = 0
    for tidsserie in range(num_tidserier):
        if compare_to_data[tidsserie] != False:
            mat_r[compare, :, 0] = data_concentration[int(compare_to_data[tidsserie]), :, 0] - sol_k[tidsserie, :, 0]
            compare += 1
    return mat_r


def calc_sum_residual(sol_k, constants, data_concentration, data_info):
    mat_r = calc_residual(sol_k, constants, data_concentration, data_info)
    sum_res = np.sum(mat_r**2)
    return sum_res


def calc_approximate_hessian(mat_a, mat_a_transpose, mat_w):
    # H = 2 * A_T * W * A
    mat_temp = np.matmul(mat_a_transpose, mat_w, axes=[(-2, -1), (-2, -1), (-2, -1)])
    mat_temp2 = np.matmul(mat_temp, mat_a, axes=[(-2, -1), (-2, -1), (-2, -1)])
    hess_approx = 2 * np.add.reduce(mat_temp2, 0)
    return hess_approx


def calc_approx_inverted_hessian(hess_approx, constants):
    '''
    try:
        inv_hess_approx = np.linalg.inv(hess_approx)
    except np.linalg.LinAlgError:
        fixed_matrix = fix_invertibility(hess_approx, constants)
        inv_hess_approx = np.linalg.inv(fixed_matrix)
    return inv_hess_approx
    '''
    eigenvalues = np.linalg.eigvals(hess_approx)
    if min(eigenvalues) <= 0:
        fixed_matrix = fix_invertibility(hess_approx, constants)
        inv_hess_approx = np.linalg.inv(fixed_matrix)
    else:
        inv_hess_approx = np.linalg.inv(hess_approx)
    return inv_hess_approx


def fix_invertibility(matrix, constants):
    num_coefficient, num_tidserier, num_tidsteg, h = constants
    indent_mat = np.identity(len(matrix))
    eigenvalues = np.linalg.eigvals(matrix)
    min_nonpos_eigenvalue = min(eigenvalues)
    gamma = - (min_nonpos_eigenvalue - h)
    fixed_matrix = matrix + gamma * indent_mat
    return fixed_matrix


def calc_descent_direction(grad, inv_hess_approx):
    p = - np.matmul(inv_hess_approx, grad)
    # p = - grad
    return p


def calc_step(p, kinetic_constants_0, sol_k, constants, data_concentration, data_info, ode_info):
    num_coefficient, num_tidserier, num_tidsteg, h = constants
    sum_res_0 = calc_sum_residual(sol_k, constants, data_concentration, data_info)
    sum_res_new = sum_res_0
    p_transpose = np.empty(num_coefficient)
    for k in range(num_coefficient):
        p_transpose[k] = p[k]
    max_p = np.amax(np.abs(p_transpose))
    p_transpose = p_transpose / max_p
    best_step = np.float64(1.0)
    stop_iteration = False
    while True:
        kinetic_constants = kinetic_constants_0 + best_step * p_transpose
        kinetic_constants[kinetic_constants < 0] = 0
        temp_sol_k, krashade = calc_sol_k(kinetic_constants, constants, ode_info)
        if krashade:
            best_step = best_step / 2
        else:
            temp_sum_res = calc_sum_residual(temp_sol_k, constants, data_concentration, data_info)
            if temp_sum_res < sum_res_0:
                sum_res_new = temp_sum_res
                break
            elif temp_sum_res >= sum_res_0:
                best_step = best_step/2
            if best_step < h:
                best_step = 0
                kinetic_constants = kinetic_constants_0
                stop_iteration = True
                break
    new_k = kinetic_constants
    return new_k, best_step, sum_res_new, stop_iteration


def start_point(k_array, constants, data_concentration, data_info, ode_info):
    print("Start koefficienter")
    print(k_array)
    sol_k_start, krashade = calc_sol_k(k_array, constants, ode_info)
    sum_res_start = calc_sum_residual(sol_k_start, constants, data_concentration, data_info)
    print("Start residue")
    print(sum_res_start)
    history = np.append(k_array, sum_res_start)
    return history


def iteration(history, constants, data_concentration, data_info, ode_info):
    k_array = history[0:-1]
    gogogo = True
    iteration_num = 1
    while gogogo:
        solution_k, krashade = calc_sol_k(k_array, constants, ode_info)
        solution_k_step = calc_sol_k_step(k_array, constants, ode_info)
        matrix_w = calc_variance(constants, data_info)
        gradient, matrix_a, matrix_a_transpose = calc_gradient(solution_k, solution_k_step, matrix_w, constants,
                                                               data_concentration, data_info)
        approximated_hessian = calc_approximate_hessian(matrix_a, matrix_a_transpose, matrix_w)
        inverted_hessian_approximation = calc_approx_inverted_hessian(approximated_hessian, constants)
        descent_direction = calc_descent_direction(gradient, inverted_hessian_approximation)
        k_array, step_length, sum_residue_0, stop_iteration = calc_step(descent_direction, k_array, solution_k,
                                                                        constants, data_concentration, data_info,
                                                                        ode_info)
        history = np.vstack((history, np.append(k_array, sum_residue_0)))
        if sum_residue_0 <= 10 ** -20:
            gogogo = False
            print("Done!")
            print("Iterations = " + str(iteration_num))
            print("Residue = " + str(sum_residue_0))
            print("Coefficients = " + str(k_array))
        if stop_iteration:
            gogogo = False
            print("Iteration stopped!")
            print("Iterations = " + str(iteration_num))
            print("Residue = " + str(sum_residue_0))
            print("Coefficients = " + str(k_array))
        if iteration_num % 50 == 0:
            print("Iterations = " + str(iteration_num))
            print("Residue = " + str(sum_residue_0))
            print("Coefficients = " + str(k_array))
        iteration_num += 1
    return history, iteration_num


def plotta_upp_yta(constants, data_concentration, data_info, ode_info, fig_and_axes):
    fig_color, ax_color = fig_and_axes
    interval_kinetic_constant_1 = np.linspace(0, 60, 100)
    interval_kinetic_constant_2 = np.linspace(0, 75, 100)
    grid_kinetic_constant_1, grid_kinetic_constant_2 = np.meshgrid(interval_kinetic_constant_1,
                                                                   interval_kinetic_constant_2)
    surface = np.empty([len(interval_kinetic_constant_1), len(interval_kinetic_constant_2)])
    for i in range(len(interval_kinetic_constant_1)):
        for o in range(len(interval_kinetic_constant_2)):
            sol_k, krashade = calc_sol_k([grid_kinetic_constant_1[i, o], grid_kinetic_constant_2[i, o]], constants,
                                         ode_info)
            if krashade:
                surface[i, o] = 999999
            else:
                surface[i, o] = calc_sum_residual(sol_k, constants, data_concentration, data_info)
    plt.figure(num=1)
    cmin = np.min(surface)
    cmax = np.max(surface)
    mesh = ax_color.pcolormesh(grid_kinetic_constant_1, grid_kinetic_constant_2, surface, vmin=cmin, vmax=cmax,
                               cmap="viridis")
    ax_color.set_xlabel(r'$\theta_1$')
    ax_color.set_ylabel(r'$\theta_2$')
    cbar = plt.colorbar(mesh, ax=ax_color)
    cbar.set_label("Belopp av kostfunktion", rotation=90, linespacing=10)


def plotta_upp_punkter(history_full, iterations, fig_and_axes):
    cb_palette = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    fig_color, ax_color = fig_and_axes
    history = np.split(history_full, iterations[0:-1])
    markers = ["o", "d", "D"]
    for gissning in range(len(history)):
        history_gissning = np.array(history[gissning])
        plt.figure(num=1)
        ax_color.scatter(history_gissning[:, 0], history_gissning[:, 1], c="black", marker=markers[gissning],
                         s=4)
    for gissning in range(len(history)):
        history_gissning = np.array(history[gissning])
        plt.figure(num=1)
        ax_color.plot(history_gissning[0, 0], history_gissning[0, 1], color=cb_palette[6], linestyle="none",
                      marker=markers[gissning])
        ax_color.plot(history_gissning[-1, 0], history_gissning[-1, 1], color=cb_palette[0], linestyle="none",
                      marker=markers[gissning])


def plotta_upp(history_full, iterations, constants, data_concentration, data_info, ode_info):
    fig_color = plt.figure(num=1)
    ax_color = plt.axes()
    fig_and_axes = [fig_color, ax_color]
    plotta_upp_yta(constants, data_concentration, data_info, ode_info, fig_and_axes)
    plotta_upp_punkter(history_full, iterations, fig_and_axes)
    save_directory = "Figurer_presentation_optimering/"
    fig_color.savefig(save_directory + "optimering_Gauss_Newton.eps", format='eps')
    plt.show()


def main():
    time_points, data_concentration = data()
    constants, ode_info, data_info = model_info(time_points)
    history_full = []
    num_iterations = []
    for gissning in range(3):
        k_array = guess_k_array(gissning)
        history = start_point(k_array, constants, data_concentration, data_info, ode_info)
        history, iterations = iteration(history, constants, data_concentration, data_info, ode_info)
        if gissning == 0:
            history_full = np.array(history)
            num_iterations = np.array([iterations])
        else:
            history_full = np.append(history_full, history, axis=0)
            num_iterations = np.append(num_iterations, num_iterations[gissning - 1] + iterations)
    plotta_upp(history_full, num_iterations, constants, data_concentration, data_info, ode_info)


main()
