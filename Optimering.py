import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd

# import read_data as rd
# import model_version as mv

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
    while gogogo:
        solution_k, krashade = calc_sol_k(k_array, constants, ode_info)
        solution_k_step = calc_sol_k_step(k_array, constants, ode_info)
        matrix_w = calc_variance(constants, data_info)
        gradient, matrix_a, matrix_a_transpose = calc_gradient(solution_k, solution_k_step, matrix_w, constants,
                                                               data_concentration, data_info)
        approximated_hessian = calc_approximate_hessian(matrix_a, matrix_a_transpose, matrix_w)
        inverted_hessian_approximation = calc_approx_inverted_hessian(approximated_hessian, constants)
        descent_direction = calc_descent_direction(gradient, inverted_hessian_approximation)
        new_k_array, step_length, sum_residue_0, stop_iteration = calc_step(descent_direction, k_array, solution_k,
                                                                            constants, data_concentration, data_info,
                                                                            ode_info)
        k_array = new_k_array
        if sum_residue_0 <= 10 ** -15:
            gogogo = False
            print("Done!")
            print("Iterations = " + str(iteration_num))
            print("Residue = " + str(sum_residue_0))
            print("Coefficients = " + str(k_array))
        if stop_iteration:
            print("Iteration stopped!")
            print("Iterations = " + str(iteration_num))
            print("Residue = " + str(sum_residue_0))
            print("Coefficients = " + str(k_array))
            break
        if iteration_num % 50 == 0:
            print("Iterations = " + str(iteration_num))
            print("Residue = " + str(sum_residue_0))
            print("Coefficients = " + str(k_array))
        iteration_num += 1


def main():
    time_points, data_concentration = data()
    constants, ode_info, data_info = model_info(time_points)
    k_array = guess_k_array()
    start_point(k_array, constants, data_concentration, data_info, ode_info)
    iteration(k_array, constants, data_concentration, data_info, ode_info)


main()

'''

bästa hittills:
Iterations = 655
Residue = 75.15890843703484
Coefficients = [0.00000000e+00 6.65716250e-01 2.88990993e+02 1.68526680e+01 5.19921204e+00 2.25612627e-01]
 
Iterations = 1259
Residue = 80.06438460092366
Coefficients = [8.19047122e+00 2.12781869e-03 3.61599002e+02 1.69504325e+01 4.08444598e-03 7.43141106e+00]

'''