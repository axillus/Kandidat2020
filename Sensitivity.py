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

from model_version import model_info, model, num_coefficient
from Optimering import calc_sol_k, calc_approximate_hessian, iteration, calc_gradient
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


#eps = np.finfo(float).eps
#_Beta = math.sqrt(eps)

def calc_S_mat(constants, ode_info, results):
    Mig1, Mig1P, SUC2, X, Krashade = calc_sol_k(results, constants, ode_info)
    num_coefficient, num_tidserier, num_tidsteg, h = constants
    t_span, t_eval, y0 = ode_info
    Kinetic_constants = results
    s_Mig1 = np.zeros((num_tidsteg, num_coefficient))
    s_Mig1P=np.zeros((num_tidsteg, num_coefficient))
    s_SUC2 = np.zeros((num_tidsteg, num_coefficient))
    s_X = np.zeros((num_tidsteg, num_coefficient))
    dy = np.zeros([num_tidserier, num_tidsteg, 1])
    for i in range(num_coefficient):
        d_Kinetic_constants = Kinetic_constants.copy()
        d_Kinetic_constants[i] = d_Kinetic_constants[i] + h
        d_solv = integrate.solve_ivp(fun=lambda t, y: model1(t, y, d_Kinetic_constants), t_span=t_span, y0=y0,
                                     method="RK45", t_eval=t_eval)
        d_solv_k = np.zeros([num_tidserier, num_tidsteg, 1])
        d_solv_k[:, :, 0] = d_solv.y
        dy_new = dy.copy()
        dy_new[:] = d_solv_k
        s_Mig1[:, i] = np.transpose((dy_new[0, :, :] - Mig1) / h)
        s_Mig1P[:, i] = np.transpose((dy_new[1, :, :] - Mig1P) / h)
        s_SUC2[:, i] = np.transpose((dy_new[2, :, :] - SUC2) / h)
        s_X[:, i] = np.transpose((dy_new[3, :, :] - X) / h)
    return s_Mig1, s_Mig1P, s_SUC2, s_X


S = np.array(calc_S_mat(Kinetic_constants))
#Model_values = np.transpose(np.array([Mig1, Mig1P, SUC2, X]))

def RMS(S, constants, ode_info, results):
    num_coefficient, num_tidserier, num_tidsteg, h = constants
    t_span, t_eval, y0 = ode_info
    Kinetic_constants = results
    Mig1, Mig1P, SUC2, X, Krashade = calc_sol_k(results, constants, ode_info)
    Model_values = np.transpose(np.array([Mig1, Mig1P, SUC2, X]))
    RMS = np.zeros((num_tidserier, len(num_coefficient)))
    S_square = np.power(S, 2)
    model_square = np.power(Model_values, 2).reshape(num_tidsteg, num_tidserier)
    for j in range(num_tidserier):
        for i in range(num_coefficient):
            K_square = np.power(Kinetic_constants, 2)
            RMS[j, i] = math.sqrt((1/num_tidsteg)*np.sum((S_square[j, :, i]*K_square[i]/model_square[:, j]), axis=0))
    return RMS


RMS = RMS(S, Kinetic_constants)