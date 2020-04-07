# Tar fram Kovariansmatrisen, variationskoefficienter samt Korrelationsmatrisen

import numpy as np
from Main import Kinetic_constants
from Main import Num_eq
from Sensitivity import S
import seaborn as sns
import matplotlib.pyplot as plt

def Var(S):
    Var_K = np.zeros((Num_eq, len(Kinetic_constants), 1))
    S_T = np.transpose(S, (0, 2, 1))
    H = 2*np.matmul(S_T, S)
    H_inv = np.linalg.inv(H)
    for i in range(Num_eq):
        c = np.sqrt(np.diag(H_inv[i, :, :]).reshape(len(Kinetic_constants), 1))
        for l in np.diag(H_inv[i]):
            if l < 0:
                print('COV ej definierad')
                break
            else:
                print('COV definierad')
        for j in range(len(Kinetic_constants)):
            Var_K[Kinetic_constants[j] == 0] = 0
            Var_K[i,j] = c[j] / Kinetic_constants[j]
    return H_inv, Var_K


Covariance, Var_K = Var(S)

def Corr(Covariance):
    v0 = np.zeros((len(Kinetic_constants), 1))
    Correlation = np.zeros((Num_eq, len(Kinetic_constants), len(Kinetic_constants)))
    for i in range(Num_eq):
        v = v0.copy()
        v = np.sqrt(np.diag(Covariance[i, :, :]))
        outer_v = np.outer(v, v)
        Correlation[i, :, :] = Covariance[i, :, :] / outer_v
    return Correlation


Correlation = Corr(Covariance)