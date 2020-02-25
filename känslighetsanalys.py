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

Theta_1=1e-5
Theta_2=2e-5
Theta_3=3e-6
Theta_4=4e-4
Beta_row=np.array([Theta_1, Theta_2, Theta_3, Theta_4])[np.newaxis]
Beta=Beta_row.transpose()
#print(Beta)
eps=np.finfo(float).eps
d_Beta=math.sqrt(eps)

N=3
M=len(Beta)
x=[[0 for i in range(M)] for j in range (N)]
A=[[0 for i in range(M)] for j in range (N)]

SUC2=x[0]
HXK1=x[1]
Mig1=x[2]
#print(x)

t=np.linspace(0,240,num=5)
#print(t)

def model(t):
    SUC2=Beta[0,:]**2*t+Beta[1,:]+Beta[2,:]*t**2+Beta[3,:]*t**3
    HXK1=Beta[0,:]+Beta[1,:]**2*t+Beta[2,:]*t+Beta[3,:]*t**2
    Mig1=Beta[0,:]*t**2+Beta[1,:]*t**3+Beta[2,:]+Beta[3,:]*t
    return [[SUC2], [HXK1], [Mig1]]
#print(model(t))

def S(t):
    Beta_new=Beta+d_Beta
    #print(Beta_new)
    SUC2 = Beta[0, :] ** 2 * t + Beta[1, :] + Beta[2, :] * t ** 2 + Beta[3, :] * t ** 3
    HXK1 = Beta[0, :] + Beta[1, :] ** 2 * t + Beta[2, :] * t + Beta[3, :] * t ** 2
    Mig1 = Beta[0, :] * t ** 2 + Beta[1, :] * t ** 3 + Beta[2, :] + Beta[3, :] * t
    for i in range(4):
        s_SUC2 = [[0 for p in range(5)] for j in range(4)]
        s_HXK1=[[0 for p in range(5)] for j in range(4)]
        s_Mig1=[[0 for p in range(5)] for j in range(4)]
        Beta[i,:]=Beta_new[i,:]
        dSUC2_row = Beta[0, :] ** 2 * t + Beta[1, :] + Beta[2, :] * t ** 2 + Beta[3, :] * t ** 3
        dSUC2=dSUC2_row.transpose()
        dHXK1_row = Beta[0, :] + Beta[1, :] ** 2 * t + Beta[2, :] * t + Beta[3, :] * t ** 2
        dHXK1=dHXK1_row.transpose()
        dMig1_row = Beta[0, :] * t ** 2 + Beta[1, :] * t ** 3 + Beta[2, :] + Beta[3, :] * t
        dMig1=dMig1_row.transpose()
        s_SUC2 += (dSUC2[:] - SUC2) / d_Beta
        s_HXK1 += (dHXK1 - HXK1) / d_Beta
        s_Mig1 += (dMig1 - Mig1) / d_Beta
    return [[s_SUC2], [s_HXK1], [s_Mig1]]
print(S(t))
#print(dSUC2_dBeta)
N=len(t)
Sensitivity_SUC2, Sensitivity_HXK1, Sensitivity_Mig1=S(t)
Sum_S=sum(np.power(Sensitivity_SUC2,2))
#RMS_s=math.sqrt((1/N))