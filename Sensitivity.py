# Vill ha känslighetsmatrisen S samt RMS
# Behöver bestämma numeriska approximationer av derivatan för alla outputs i särskilda tidpunkter t_i över paramater p
# Behöver outputs, y_i
# Behöver tidsstegning
# Behöver parametrar, p_i
# Definiera steglängd av parametrar
# Definiera partiella derivator i S
# Bestäm RMS mha S

import numpy as np
import scipy.integrate as integrate
import math

from read_data import data
from model_version import model_info, model, model_1, model_1_k6_k7, model_1_k5_k3
from Plot_optimering import read_results, get_min_cost


def calc_sol_k():
    time_points, data_conc = data()
    results = read_results()
    constants, ode_info, data_info = model_info(time_points)
    vald_modell, num_coefficient, num_tidserier, num_tidsteg, h = constants
    best_coefficients, min_cost_funk, min_viktad_cost_funk = get_min_cost(results)
    t_span, t_eval, y0 = ode_info
    sol = integrate.solve_ivp(fun=lambda t, y: model(vald_modell, t, y, best_coefficients), t_span=t_span, y0=y0, method="RK45",
                              t_eval=t_eval)
    sol_k = np.empty([num_tidserier, num_tidsteg, 1])
    krashade = False
    try:
        sol_k[:, :, 0] = sol.y
    except ValueError:
        krashade = True
    return sol_k, krashade


def calc_S_mat():
    time_points, data_conc = data()
    results = read_results()
    constants, ode_info, data_info = model_info(time_points)
    vald_modell, num_coefficient, num_tidserier, num_tidsteg, h = constants
    sol_k, krashade = calc_sol_k()
    suc2, mig1, mig1_phos, X = sol_k
    t_span, t_eval, y0 = ode_info
    best_coefficients, min_cost_funk, min_viktad_cost_funk = get_min_cost(results)
    Kinetic_constants = best_coefficients
    s_suc2 = np.zeros((num_tidsteg, num_coefficient))
    s_mig1 = np.zeros((num_tidsteg, num_coefficient))
    s_mig1_phos = np.zeros((num_tidsteg, num_coefficient))
    s_X = np.zeros((num_tidsteg, num_coefficient))
    dy = np.zeros([num_tidserier, num_tidsteg, 1])
    for i in range(num_coefficient):
        d_Kinetic_constants = Kinetic_constants.copy()
        d_Kinetic_constants[i] = d_Kinetic_constants[i] + h
        d_solv = integrate.solve_ivp(fun=lambda t, y: model(vald_modell, t, y, d_Kinetic_constants), t_span=t_span, y0=y0,
                                     method="RK45", t_eval=t_eval)
        d_solv_k = np.zeros([num_tidserier, num_tidsteg, 1])
        d_solv_k[:, :, 0] = d_solv.y
        dy_new = dy.copy()
        dy_new[:] = d_solv_k
        s_suc2[:, i] = np.transpose((dy_new[0] - suc2) / h)
        s_mig1[:, i] = np.transpose((dy_new[1] - mig1) / h)
        s_mig1_phos[:, i] = np.transpose((dy_new[2] - mig1_phos) / h)
        s_X[:, i] = np.transpose((dy_new[3] - X) / h)
    S = np.array([s_suc2, s_mig1, s_mig1_phos, s_X])
    return S


def RMS():
    S = calc_S_mat()
    time_points, data_conc = data()
    results = read_results()
    constants, ode_info, data_info = model_info(time_points)
    vald_modell, num_coefficient, num_tidserier, num_tidsteg, h = constants
    best_coefficients, min_cost_funk, min_viktad_cost_funk = get_min_cost(results)
    Kinetic_constants = best_coefficients
    sol_k, krashade = calc_sol_k()
    suc2, mig1, mig1_phos, X = sol_k
    Model_values = np.transpose(np.array([suc2, mig1, mig1_phos, X]))
    rms = np.zeros((2, num_coefficient))
    S_square = np.power(S, 2)
    model_square = np.power(Model_values, 2).reshape(num_tidsteg, num_tidserier)
    for j in range(2):
        for i in range(num_coefficient):
            K_square = np.power(Kinetic_constants, 2)
            rms[j, i] = math.sqrt((1/num_tidsteg)*np.sum((S_square[j, :, i]*K_square[i]/model_square[:, j]), axis=0))
    return rms


def save_RMS():
    rms = RMS()
    np.savetxt('RMS_suc2_k3k5', rms[0, :])
    np.savetxt('RMS_mig1_k3k5', rms[1, :])
