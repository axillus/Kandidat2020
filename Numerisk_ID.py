# Tar fram Kovariansmatrisen, variationskoefficienter samt Korrelationsmatrisen

import numpy as np

#from Main import Kinetic_constants
#from Main import Num_eq
from Sensitivity import S
from model_version import model_info, num_coefficient
from Optimering import iteration

def Var(S, constants, results):
    num_coefficient, num_tidserier, num_tidsteg, h = constants
    Kinetic_constants = results
    Var_K = np.zeros((num_tidserier, num_coefficient, 1))
    S_T = np.transpose(S, (0, 2, 1))
    H = 2*np.matmul(S_T, S)
    H_inv = np.linalg.inv(H)
    for i in range(num_tidserier):
        c = np.sqrt(np.diag(H_inv[i, :, :]).reshape(num_coefficient, 1))
        for l in np.diag(H_inv[i]):
            if l < 0:
                print('COV ej definierad')
                break
        for j in range(num_coefficient):
            Var_K[Kinetic_constants[j] == 0] = 0
            Var_K[i,j] = c[j] / Kinetic_constants[j]
    return H_inv, Var_K


Covariance, Var_K = Var(S)

def save_m(Covariance, Var_K):
    np.savetxt('Cov_Mig1', Covariance[0,:,:])
    np.savetxt('Cov_SUC2', Covariance[2,:,:])
    np.savetxt('Var_K_Mig1', Var_K[0, :, :])
    np.savetxt('Var_K_SUC2', Var_K[2, :, :])


def Corr(Covariance, constants):
    num_coefficient, num_tidserier, num_tidsteg, h = constants
    v0 = np.zeros((num_coefficient, 1))
    Correlation = np.zeros((Num_eq, num_coefficient, num_coefficient))
    for i in range(num_tidserier):
        v = v0.copy()
        v = np.sqrt(np.diag(Covariance[i, :, :]))
        outer_v = np.outer(v, v)
        Correlation[i, :, :] = Covariance[i, :, :] / outer_v
    return Correlation


Correlation = Corr(Covariance)
