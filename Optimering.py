import numpy as np
import scipy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import pandas as pd

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


def model1(t, y, kinetic_constants):
    suc2 = y[0]
    mig1 = y[1]
    mig1_phos = y[2]
    X = y[3]
    r1 = kinetic_constants[0] * mig1_phos
    r2 = mig1 * kinetic_constants[1]
    r3 = kinetic_constants[2]/mig1
    r4 = suc2*kinetic_constants[3]
    r5 = kinetic_constants[4]/mig1
    r6 = X * kinetic_constants[5]
    dmig1_dt = r1 - r2 + r6
    dmig_phos_dt = - r1 + r2
    dsuc2_dt = r3 - r4
    dX_dt = r5 - r6
    dy_dt = [dsuc2_dt, dmig1_dt, dmig_phos_dt, dX_dt]
    return dy_dt


def calc_sol_k(kinetic_constants_0):
    sol = integrate.solve_ivp(fun=lambda t, y: model1(t, y, kinetic_constants_0), t_span=t_span, y0=y0, method="RK45",
                              t_eval=t_eval)
    sol_k = np.empty([num_tidserier, num_tidsteg, 1])
    sol_k[:, :, 0] = sol.y
    # shape = (tidsserie(3), tidssteg(100), värde(1))
    return sol_k


def calc_sol_k_step(kinetic_constants_0):
    sol_k_step = np.empty([num_tidserier, num_tidsteg, num_coefficient])
    for k in range(num_coefficient):
        kinetic_constants_step = kinetic_constants_0.copy()
        kinetic_constants_step[k] = kinetic_constants_step[k] + h
        sol = integrate.solve_ivp(fun=lambda t, y: model1(t, y, kinetic_constants_step), t_span=t_span, y0=y0,
                                  method="RK45", t_eval=t_eval)
        sol_k_step[:, :, k] = sol.y
    # shape = (tidsserie(3), tidssteg(100), värde i k-steg(4))
    return sol_k_step


def calc_variance():
    W = np.empty([num_tidserier, num_tidsteg, num_tidsteg])
    for tidserie in range(num_tidserier):
        W[tidserie, :, :] = np.identity(num_tidsteg)
    return W


def calc_gradient(sol_k, sol_k_step, W):
    mat_r = calc_residual(sol_k)
    diff_k_t = np.empty([num_tidserier, num_tidsteg, num_coefficient])
    mat_A = np.empty([num_tidserier, num_tidsteg, num_coefficient])  # mat_A = S_T, S = sensitivity matrix
    for k in range(num_coefficient):
        diff_k_t[:, :, k] = sol_k_step[:, :, k] - solution_k[:, :, 0]
        mat_A[:, :, k] = diff_k_t[:, :, k]/h
    mat_A_T = np.transpose(mat_A, (0, 2, 1))  # mat_A_T = S, S = sensitivity matrix
    # A.shape = (3, 100, 4)
    # A_T.shape = (3, 4, 100)
    # r.shape = (3, 100, 1)
    # A_T.shape = (3, 4, 100)
    # W.shape = (3, 100, 100)

    # grad = - 2 * A_T * W * r

    mat_temp = np.matmul(mat_A_T, W, axes=[(-2, -1), (-2, -1), (-2, -1)])
    # mat_temp.shape = (3, 4, 100)
    mat_temp2 = np.matmul(mat_temp, mat_r, axes=[(-2, -1), (-2, -1), (-2, -1)])
    # mat_temp2.shape = (3, 4, 1)
    grad = -2 * np.add.reduce(mat_temp2, 0)
    # grad.shape = (4, 1)
    return grad, mat_A  # skicka med mat_A_T ?


def calc_residual(sol_k):
    mat_r = np.empty([num_tidserier, num_tidsteg, 1])
    for tidsserie in range(3):
        mat_r[tidsserie, :, 0] = data_concentration[tidsserie, :, 0] - sol_k[tidsserie, :, 0]
    return mat_r


def calc_sum_residual(sol_k):
    mat_r = calc_residual(sol_k)
    sum_res = np.sum(mat_r**2)
    return sum_res


def calc_approximate_hessian(mat_A, mat_W):
    mat_A_T = np.transpose(mat_A, (0, 2, 1))
    # H = 2 * A_T * W * A
    mat_temp = np.matmul(mat_A_T, mat_W, axes=[(-2, -1), (-2, -1), (-2, -1)])
    # mat_temp.shape = (3, 4, 100)
    mat_temp2 = np.matmul(mat_temp, mat_A, axes=[(-2, -1), (-2, -1), (-2, -1)])
    # mat_temp2.shape = (3, 4, 4)
    hess_approx = 2 * np.add.reduce(mat_temp2, 0)
    # hess_approx.shape = (4, 4)
    return hess_approx


def calc_approx_inverted_hessian(hess_approx):
    inv_hess_approx = np.linalg.inv(hess_approx)
    #inv_hess_approx = np.linalg.pinv(hess_approx)
    return inv_hess_approx


def calc_descent_direction(grad, inv_hess_approx):
    p = - np.matmul(inv_hess_approx, grad)
    return p


def calc_step(p, kinetic_constants_0, sol_k):
    low_sum_res = calc_sum_residual(sol_k)
    p_transpose = np.empty(num_coefficient)
    for k in range(num_coefficient):
        p_transpose[k] = p[k]
    best_step = eps
    for new_step_length in np.linspace(eps, 1, 100):
        kinetic_constants = kinetic_constants_0 + new_step_length*p_transpose
        temp_sol_k = calc_sol_k(kinetic_constants)
        temp_sum_res = calc_sum_residual(temp_sol_k)
        if temp_sum_res < low_sum_res:
            best_step = new_step_length
        elif temp_sum_res > low_sum_res:
            break
    new_k = kinetic_constants_0 + best_step * p_transpose
    return new_k, best_step


def calc_step2(p, kinetic_constants_0, sol_k):
    sum_res_0 = calc_sum_residual(sol_k)
    p_transpose = np.empty(num_coefficient)
    for k in range(num_coefficient):
        p_transpose[k] = p[k]
    best_step = 1
    while True:
        kinetic_constants = kinetic_constants_0 + best_step * p_transpose
        temp_sol_k = calc_sol_k(kinetic_constants)
        temp_sum_res = calc_sum_residual(temp_sol_k)
        if temp_sum_res < sum_res_0:
            break
        elif temp_sum_res > sum_res_0:
            best_step = best_step/2
    new_k = kinetic_constants_0 + best_step * p_transpose
    return new_k, best_step


def main():
    num_coefficient = 4
    K_array = np.array([1, 1, 1, 1, 1], np.float64)
    num_tidserier = 4
    t_eval = np.linspace(0, 1, num=100)
    num_tidsteg = len(t_eval)
    eps = np.finfo(float).eps
    h = np.sqrt(eps)
    t_span = [0, 1]
    SUC2_0 = 1
    MIG1_0 = 1
    MIG1_PHOS_0 = 1
    X_0 = 1
    y0 = np.array([SUC2_0, MIG1_0, MIG1_PHOS_0, X_0])

    data_concentration = data()

    gogogo = True
    iteration = 0
    while gogogo:
        solution_k = calc_sol_k(K_array)
        solution_k_step = calc_sol_k_step(K_array)
        matrix_W = calc_variance()
        gradient, matrix_A = calc_gradient(solution_k, solution_k_step, matrix_W)
        approximated_hessian = calc_approximate_hessian(matrix_A, matrix_W)
        inverted_hessian_approximation1, inverted_hessian_approximation2 = calc_approx_inverted_hessian(
            approximated_hessian)
        descent_direction = calc_descent_direction(gradient, inverted_hessian_approximation1,
                                                   inverted_hessian_approximation2)
        new_K_array, step_length = calc_step2(descent_direction, K_array, solution_k)
        K_array = new_K_array
        print(step_length)
        print(K_array)
        sum_residue_0 = calc_sum_residual(solution_k)
        if sum_residue_0 <= 10 ** -14:
            gogogo = False
            print("Done!")
            print("Iterations = " + str(iteration))
            print("Residue = " + str(sum_residue_0))
            print("Coefficients = " + str(K_array))
        iteration = iteration + 1
        if iteration % 1000 == 0:
            print("Iterations = " + str(iteration))
            print("Residue = " + str(sum_residue_0))
            print("Coefficients = " + str(K_array))


main()


