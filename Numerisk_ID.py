# Tar fram Kovariansmatrisen, variationskoefficienter samt Korrelationsmatrisen

import numpy as np

from read_data import data
from model_version import model_info, model
from Plot_optimering import read_results, get_min_cost
from Sensitivity import calc_S_mat


def h_inverse():
    S = calc_S_mat()
    S_T = np.transpose(S, (0, 2, 1))
    H = 2*np.matmul(S_T, S)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        fixed_matrix = fix_invertibility(H)
        H_inv = np.linalg.inv(fixed_matrix)
    return H_inv


def fix_invertibility(matrix):
    time_points, data_conc = data()
    constants, ode_info, data_info = model_info(time_points)
    vald_modell, num_coefficient, num_tidserier, num_tidsteg, h = constants
    indent_mat = np.identity(num_coefficient)
    eigenvalues = np.linalg.eigvals(matrix)
    min_nonpos_eigenvalue = np.amin(eigenvalues)
    gamma = - (min_nonpos_eigenvalue - h)
    fixed_matrix = matrix + gamma * indent_mat
    return fixed_matrix


def var_K():
    H_inv = h_inverse()
    time_points, data_conc = data()
    constants, ode_info, data_info = model_info(time_points)
    vald_modell, num_coefficient, num_tidserier, num_tidsteg, h = constants
    results = read_results(constants)
    best_coefficients, min_cost_funk, min_viktad_cost_funk = get_min_cost(results)
    Kinetic_constants = best_coefficients
    Var_K = np.zeros((num_tidserier, num_coefficient, 1))
    for i in range(num_tidserier):
        c = np.sqrt(np.diag(H_inv[i, :, :]).reshape(num_coefficient, 1))
        for l in np.diag(H_inv[i]):
            if l < 0:
                print('COV ej definierad')
                break
        for j in range(num_coefficient):
            if Kinetic_constants[j] == 0:
                Var_K[i,j] = 0
            else:
                Var_K[i,j] = c[j] / Kinetic_constants[j]
    return Var_K


def save_m():
    H_inv = h_inverse()
    Covariance = H_inv
    Var_K = var_K()
    np.savetxt('Cov_mig1_k6k7', Covariance[1,:,:])
    np.savetxt('Cov_suc2_k6k7', Covariance[0,:,:])
    np.savetxt('Var_K_mig1_k6k7', Var_K[1, :, :])
    np.savetxt('Var_K_suc2_k6k7', Var_K[0, :, :])


def corr():
    H_inv = h_inverse()
    Covariance = H_inv
    time_points, data_conc = data()
    constants, ode_info, data_info = model_info(time_points)
    vald_modell, num_coefficient, num_tidserier, num_tidsteg, h = constants
    v0 = np.zeros((num_coefficient, 1))
    Correlation = np.zeros((num_tidserier, num_coefficient, num_coefficient))
    for i in range(num_tidserier):
        v = v0.copy()
        v = np.sqrt(np.diag(Covariance[i, :, :]))
        outer_v = np.outer(v, v)
        Correlation[i, :, :] = Covariance[i, :, :] / outer_v
    return Correlation


def save_corr():
    Correlation = corr()
    np.savetxt('Cor_mig1_k6k7', Correlation[1, :, :])
    np.savetxt('Cor_suc2_k6k7', Correlation[0, :, :])
