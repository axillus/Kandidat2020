import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd
import csv

from read_data import data
from model_version import model, model_info, guess_k_array

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


def calc_gradient(sol_k, sol_k_step, constants, data_concentration, data_info):
    # grad = - 2 * A_T * r
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
    mat_temp = np.matmul(mat_a_transpose, mat_r, axes=[(-2, -1), (-2, -1), (-2, -1)])
    grad = -2 * np.add.reduce(mat_temp, 0)
    return grad, mat_a, mat_a_transpose


def calc_residual(sol_k, constants, data_concentration, data_info):
    num_coefficient, num_tidserier, num_tidsteg, h = constants
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


def calc_sum_residual(sol_k, constants, data_concentration, data_info):
    mat_r = calc_residual(sol_k, constants, data_concentration, data_info)
    sum_res = np.sum(mat_r**2)
    return sum_res


def calc_approximate_hessian(mat_a, mat_a_transpose):
    # H = 2 * A_T * A
    mat_temp = np.matmul(mat_a_transpose, mat_a, axes=[(-2, -1), (-2, -1), (-2, -1)])
    hess_approx = 2 * np.add.reduce(mat_temp, 0)
    return hess_approx


def calc_approx_inverted_hessian(hess_approx, constants):
    try:
        inv_hess_approx = np.linalg.inv(hess_approx)
    except np.linalg.LinAlgError:
        fixed_matrix = fix_invertibility(hess_approx, constants)
        inv_hess_approx = np.linalg.inv(fixed_matrix)
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


def iteration(k_array, constants, data_concentration, data_info, ode_info):
    gogogo = True
    iteration_num = 1
    results = np.append(k_array, np.inf)
    while gogogo:
        solution_k, krashade = calc_sol_k(k_array, constants, ode_info)
        solution_k_step = calc_sol_k_step(k_array, constants, ode_info)
        gradient, matrix_a, matrix_a_transpose = calc_gradient(solution_k, solution_k_step, constants,
                                                               data_concentration, data_info)
        approximated_hessian = calc_approximate_hessian(matrix_a, matrix_a_transpose)
        inverted_hessian_approximation = calc_approx_inverted_hessian(approximated_hessian, constants)
        descent_direction = calc_descent_direction(gradient, inverted_hessian_approximation)
        k_array, step_length, sum_residue_0, stop_iteration = calc_step(descent_direction, k_array, solution_k,
                                                                        constants, data_concentration, data_info,
                                                                        ode_info)
        if sum_residue_0 <= 10 ** -15:
            gogogo = False
            print("Done!")
            print("Iterations = " + str(iteration_num))
            print("Residue = " + str(sum_residue_0))
            print("Coefficients = " + str(k_array))
            results = np.append(k_array, sum_residue_0)
        if stop_iteration:
            print("Iteration stopped!")
            print("Iterations = " + str(iteration_num))
            print("Residue = " + str(sum_residue_0))
            print("Coefficients = " + str(k_array))
            results = np.append(k_array, sum_residue_0)
            break
        if iteration_num % 50 == 0:
            print("Iterations = " + str(iteration_num))
            print("Residue = " + str(sum_residue_0))
            print("Coefficients = " + str(k_array))
        iteration_num += 1
    return results


def save_results(results):
    with open("fixad_model_1.csv", "a") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=",")
        csvWriter.writerow(results)


def main():
    time_points, data_concentration = data()
    constants, ode_info, data_info = model_info(time_points)
    for i in range(1000):
        k_array = guess_k_array()
        start_point(k_array, constants, data_concentration, data_info, ode_info)
        results = iteration(k_array, constants, data_concentration, data_info, ode_info)
        save_results(results)


main()


