#Vill ha känslighetsmatrisen S samt RMS
#Behöver bestämma numeriska approximationer av derivatan för alla outputs i särskilda tidpunkter t_i över paramater p
#Behöver outputs, y_i
#Behöver tidsstegning
#Behöver parametrar, p_i
#Definiera steglängd av parametrar
#Definiera partiella derivator i S
#Bestäm RMS mha S

import numpy as np
import math
from Model1 import sol()
from Main import Kinetic_constants
from Main import y0
from Main import t_span
from Main import t_eval
from Main import Num_eq
from Main import Num_timestep

SUC2, Mig1, Mig1P, X=sol()
eps = np.finfo(float).eps
d_Beta = math.sqrt(eps)

def calc_S_mat(Kinetic_constants):
    s_Mig1 = np.zeros((Num_timestep, len(Kinetic_constants)))
    s_Mig1_P=np.zeros((Num_timestep, len(Kinetic_constants)))
    s_SUC2 = np.zeros((Num_timestep, len(Kinetic_constants)))
    s_X = np.zeros((Num_timestep, len(Kinetic_constants)))
    dy = np.zeros([Num_eq,Num_timestep,1])
    for i in range(len(Kinetic_constants)):
        d_Kinetic_constants = Kinetic_constants.copy()
        d_Kinetic_constants[i] = d_Kinetic_constants[i] + d_Beta
        d_solv = integrate.solve_ivp(fun=lambda t, y: simp_model(t, y, d_Kinetic_constants), t_span=t_span, y0=y0,
                                   method="RK45",t_eval=t_eval)
        d_solv_k = np.zeros([Num_eq, Num_timestep, 1])
        d_solv_k[:, :, 0] = d_solv.y
        dy_new=dy.copy()
        dy_new[:]=d_solv_k
        s_Mig1[:,i] = np.transpose((dy_new[0,:,:] - Mig1) / d_Beta)
        s_Mig1_P[:,i] = np.transpose((dy_new[1,:,:] - Mig1_P) / d_Beta)
        s_SUC2[:, i] = np.transpose((dy_new[2, :, :] - SUC2) / d_Beta)
        s_X[:,i] = np.transpose((dy_new[3, :, :] - X) / d_Beta)
    return Mig1, Mig1_P, SUC2, X

S=np.array(calc_S_mat(Kinetic_constants))
Model_values=np.transpose(np.array([Mig1, Mig1_P, SUC2, X]))

def RMS(S,Kinetic_constants):
    RMS=np.zeros((4,len(Kinetic_constants)))
    S_square=np.power(S,2)
    model_square=np.power(Model_values,2).reshape(2,2)
    for j in range(4):
        for i in range(len(Kinetic_constants)):
            K_square=np.power(Kinetic_constants,2)
            RMS[j,i]=math.sqrt((1/len(t_eval))*np.sum((S_square[j,:,i]*K_square[i]/model_square[:,j]),axis=0))
    return RMS

RMS=RMS(S,Kinetic_constants)