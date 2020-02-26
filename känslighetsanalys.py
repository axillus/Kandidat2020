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

Theta_1=1e-5
Theta_2=2e-5
Theta_3=3e-6
Theta_4=4e-4
Kinetic_constants_row=np.array([Theta_1, Theta_2, Theta_3, Theta_4])[np.newaxis]
Kinetic_constants=np.transpose(Kinetic_constants_row)
Kinetic_constants_0=np.zeros((4,1))
#print(len(Kinetic_constants))

Koefficient_M=np.empty([3,len(Kinetic_constants)])
Koefficient_M=[[1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 1]]
#print(Koefficient_M.shape)

#SUC2 = y[0]
#HXK1 = y[1]
#Mig1 = y[2]

eps=np.finfo(float).eps
d_Beta=math.sqrt(eps)

#N=3
#M=len(Beta)
#y=[[0 for i in range(M)] for j in range (N)]
#A=[[0 for i in range(M)] for j in range (N)]
#print(x)

t_col=np.linspace(0,240,num=5, dtype=int)
t=np.transpose(t_col)

y=np.empty([3,len(t)])
def model(t, y, Kinetic_constants):
    y=np.empty([3,len(t)])
    A_mat=np.matmul(Koefficient_M,Kinetic_constants)
    y[0]=A_mat[0]*t
    y[1]=A_mat[1]*t
    y[2]=A_mat[2]*t
    return [y]
print(model(t,y,Kinetic_constants))

def calc_S_mat(Kinetic_constants,t):
    s_SUC2=np.empty((4,5))
    s_HXK1=np.empty((4,5))
    s_Mig1=np.empty((4,5))
    dy = np.zeros([3, len(t)])
    A_mat = np.matmul(Koefficient_M, Kinetic_constants)
    y[0] = A_mat[0] * t
    print(y[0])
    y[1] = A_mat[1] * t
    y[2] = A_mat[2] * t
    for i in range(len(Kinetic_constants)):
        d_Kinetic_constants=Kinetic_constants_0.copy()
        d_Kinetic_constants[i]=d_Kinetic_constants[i]+d_Beta
        A_new_mat=np.matmul(Koefficient_M,d_Kinetic_constants)
        dy[0] = np.transpose(A_new_mat[0] * t)
        dy[1] = np.transpose(A_new_mat[1] * t)
        dy[2] = np.transpose(A_new_mat[2] * t)
        s_SUC2[i] += (dy[0,:] - y[0,:]) / d_Beta
        s_HXK1[i] += (dy[1,:] - y[1,:]) / d_Beta
        s_Mig1[i] += (dy[2,:] - y[2,:]) / d_Beta
        #print(s_SUC2)
    return [[s_SUC2], [s_HXK1], [s_Mig1]]
#print(calc_S_mat(Kinetic_constants,t))

N=len(t)
Sensitivity_SUC2, Sensitivity_HXK1, Sensitivity_Mig1=calc_S_mat(Kinetic_constants,t)
#p/SUC2, p/HXK1, p/Mig1=model(t,y,Kinetic_constants)
Sum_S=sum(np.power(Sensitivity_SUC2,2)*np.power((Kinetic_constants/y[0]),2))
RMS_s=math.sqrt((1/N))

