#Vill ha känslighetsmatrisen S och RMS(s_ij)
#Behöver bestämma numeriska approximationer av derivatan för alla outputs i särskilda tidpunkter t_i över paramater p
#Definiera outputs, y_i
#Definiera tidsstegning
#Definiera parametrar, p_i
#Definiera steglängd av parametrar
#Definiera partiella derivator i S
#Ta fram RMS
import numpy as np
import math
import pandas as pd

Theta_1 = 1e-5
Theta_2 = 2e-5
Theta_3 = 3e-6
Theta_4 = 4e-4
Kinetic_constants_row = np.array([Theta_1, Theta_2, Theta_3, Theta_4])[np.newaxis]
Kinetic_constants = np.transpose(Kinetic_constants_row)
Kinetic_constants_0 = np.zeros((4, 1))
# print(Kinetic_constants)

Koefficient_M = np.empty([3, len(Kinetic_constants)])
Koefficient_M = [[1, 1, 0.5, 0], [1, 0.75, 0, 1], [0, 1, 1, 1]]
# print(Koefficient_M.shape)

# SUC2 = y[0]
# HXK1 = y[1]
# Mig1 = y[2]

eps = np.finfo(float).eps
d_Beta = math.sqrt(eps)


t_col = np.linspace(0, 240, num=5, dtype=int)
t = np.transpose(t_col)

y = np.empty([3, len(t)])


def model(t, Kinetic_constants):
    y = np.empty([3, len(t)])
    A_mat = np.matmul(Koefficient_M, Kinetic_constants)
    y = A_mat * t
    return [y[0], y[1], y[2]]


SUC2, HXK1, Mig1 = model(t, Kinetic_constants)
print(SUC2)


def calc_S_mat(Kinetic_constants, t):
    s_SUC2 = np.zeros((4, 5))
    s_HXK1 = np.zeros((4, 5))
    s_Mig1 = np.zeros((4, 5))
    dy = np.empty([3, len(t)])
    for i in range(len(Kinetic_constants)):
        d_Kinetic_constants = Kinetic_constants.copy()
        # print(d_Kinetic_constants)
        d_Kinetic_constants[i, :] = d_Kinetic_constants[i] + d_Beta
        # print(d_Kinetic_constants)
        A_new_mat = np.matmul(Koefficient_M, d_Kinetic_constants)
        # print(A_new_mat)
        dy = A_new_mat * t
        # print(dy)
        s_SUC2[i] += (dy[0, :] - SUC2) / d_Beta
        s_HXK1[i] += (dy[1, :] - HXK1) / d_Beta
        s_Mig1[i] += (dy[2, :] - Mig1) / d_Beta
        # print(s_SUC2)
    return [s_SUC2, s_HXK1, s_Mig1]


# print(calc_S_mat(Kinetic_constants,t))


N = len(t)
Sensitivity_SUC2, Sensitivity_HXK1, Sensitivity_Mig1 = calc_S_mat(Kinetic_constants, t)


# p/SUC2, p/HXK1, p/Mig1=model(t,y,Kinetic_constants)
# print(Sensitivity_SUC2)


def RMS_SUC2(Kinetic_constants):
    RMS_s = np.zeros((1, 4))
    K_0 = np.zeros((4, 1))
    S_square = np.power(Sensitivity_SUC2, 2)
    Sum_S_square = np.sum(S_square, axis=1)
    Sum_S = np.sum(SUC2, axis=0)
    # print(Sum_S_square[0])
    for i in range(len(Kinetic_constants)):
        K = K_0.copy()
        K[i] = Kinetic_constants[i] / Sum_S
        K_square = np.power(K, 2)
        # print(K_square)
        S = (1 / N) * Sum_S_square[i] * K_square[i]
        # print(S)
        RMS_s[:, i] += math.sqrt(S)
        # print(RMS_s)
    return [RMS_s]


print(RMS_SUC2(Kinetic_constants))

def Variations_koefficient(Sensitivity):
    S = Sensitivity.reshape(4, 15)
   # print(S)
    S_T=np.transpose(S)
    #print(S_T.shape)
    H=2*np.matmul(S,S_T)
   # print(H)
    H_inv=np.linalg.inv(H)
    C=np.diag(H_inv)
    for i in range(len(Kinetic_constants)):
        Var_K=C/Kinetic_constants[i]
    return Var_K
#print(Variations_koefficient(Sensitivity))