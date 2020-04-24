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

from model_version import model_info, model, num_coeff
#from Optimering import calc_sol_k
from Plot_optimering import read_results, get_min_cost

#from Model1 import Mig1P
#from Model1 import SUC2
#from Model1 import X
#from Main import Kinetic_constants
#from Main import y0
#from Main import t_span
#from Main import t_eval
#from Main import Num_eq
#from Main import Num_timestep
#from Main import model1

def calc_sol_k(results, constants, ode_info):
    results = read_results()
    num_coefficient, num_tidserier, num_tidsteg, h = constants
    Kinetic_constants, min_cost = get_min_cost(results)
    t_span, t_eval, y0 = ode_info
    sol = integrate.solve_ivp(fun=lambda t, y: model(t, y, Kinetic_constants), t_span=t_span, y0=y0, method="RK45",
                              t_eval=t_eval)
    sol_k = np.empty([num_tidserier, num_tidsteg, 1])
    krashade = False
    try:
        sol_k[:, :, 0] = sol.y
    except ValueError:
        krashade = True
    return sol_k, krashade


def calc_S_mat(ode_info, constants, results):
    results = read_results()
    num_coefficient, num_tidserier, num_tidsteg, h = constants
    suc2, mig1, mig1_phos, X, Krashade = calc_sol_k(best_coefficient, constants, ode_info)
    t_span, t_eval, y0 = ode_info
    Kinetic_constants, min_cost = get_min_cost(results)
    s_suc2 = np.zeros((num_tidsteg, num_coefficient))
    s_mig1 = np.zeros((num_tidsteg, num_coefficient))
    s_mig1_phos=np.zeros((num_tidsteg, num_coefficient))
    s_X = np.zeros((num_tidsteg, num_coefficient))
    dy = np.zeros([num_tidserier, num_tidsteg, 1])
    for i in range(num_coefficient):
        d_Kinetic_constants = Kinetic_constants.copy()
        d_Kinetic_constants[i] = d_Kinetic_constants[i] + h
        d_solv = integrate.solve_ivp(fun=lambda t, y: model(t, y, d_Kinetic_constants), t_span=t_span, y0=y0,
                                     method="RK45", t_eval=t_eval)
        d_solv_k = np.zeros([num_tidserier, num_tidsteg, 1])
        d_solv_k[:, :, 0] = d_solv.y
        dy_new = dy.copy()
        dy_new[:] = d_solv_k
        s_suc2[:, i] = np.transpose((dy_new[0, :, :] - suc2) / h)
        s_mig1[:, i] = np.transpose((dy_new[1, :, :] - mig1) / h)
        s_mig1_phos[:, i] = np.transpose((dy_new[2, :, :] - mig1_phos) / h)
        s_X[:, i] = np.transpose((dy_new[3, :, :] - X) / h)
        S = np.array([s_suc2, s_mig1, s_mig1_phos, s_X])
    return S


def RMS(S, constants, results):
    S = calc_S_mat(ode_info, constants, results)
    results = read_results()
    num_coefficient, num_tidserier, num_tidsteg, h = constants
    Kinetic_constants, min_cost = get_min_cost(results)
    suc2, mig1, mig1_phos, X, Krashade = calc_sol_k(best_coefficient, constants, ode_info)
    Model_values = np.transpose(np.array([suc2, mig1, mig1_phos, X]))
    RMS = np.zeros((num_tidserier, num_coefficient))
    S_square = np.power(S, 2)
    model_square = np.power(Model_values, 2).reshape(num_tidsteg, num_tidserier)
    for j in range(num_tidserier):
        for i in range(num_coefficient):
            K_square = np.power(Kinetic_constants, 2)
            RMS[j, i] = math.sqrt((1/num_tidsteg)*np.sum((S_square[j, :, i]*K_square[i]/model_square[:, j]), axis=0))
            print(RMS)
    return RMS


def save_RMS(RMS):
    np.savetxt('RMS_suc2', RMS[0,:,:])
    np.savetxt('RMS_mig1', RMS[1,:,:])