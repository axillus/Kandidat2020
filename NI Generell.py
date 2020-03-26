import numpy as np
import math
import scipy.integrate as integrate

Kinetic_constants = [k1, k2, k2...]
t_span = np.linspace(0, 240, num=100)
y0 = [y01, y02]
eps = np.finfo(float).eps
d_Beta = math.sqrt(eps)
Num_eq=#Ekvationer
Num_timestep=#tidssteg

def simp_model(t,y,Kinetic_constants):
    SUC2=y[0]
    Mig1=y[1]
    Mig1_P=y[2]
    r1=Kinetic_constants[0]*A
    r2=Kinetic_constants[1]*B
    r3=...
    dSUC2_dt=-r1+r2
    dMig1_dt=-r2+r1
    dy_dt=[dSUC2_dt, dMig1_dt]
    return dy_dt


def sol():
    solv=integrate.solve_ivp(fun=lambda t, y: simp_model(t, y, Kinetic_constants), t_span=t_span, y0=y0, method="RK45")
    solv_k = np.zeros([Num_eq, Num_timestep, 1])
    solv_k[:, :, 0] = solv.y
    return solv_k

SUC2, Mig1, Mig1P=sol()


def calc_S_mat():
    s_SUC2= np.zeros((Num_timestep, len(Kinetic_constants)))
    s_Mig1 = np.zeros((Num_timestep, len(Kinetic_constants)))
    s_Mig1_P=np.zeros((Num_timestep, len(Kinetic_constants)))
    dy = np.zeros([Num_eq,Num_timestep,1])
    for i in range(len(Kinetic_constants)):
        d_Kinetic_constants = Kinetic_constants.copy()
        d_Kinetic_constants[i] = d_Kinetic_constants[i] + d_Beta
        d_solv = integrate.solve_ivp(fun=lambda t, y: simp_model(t, y, d_Kinetic_constants), t_span=t_span, y0=y0,
                                   method="RK45")
        d_solv_k = np.zeros([Num_eq, Num_timestep, 1])
        d_solv_k[:, :, 0] = d_solv.y
        dy_new=dy.copy()
        dy_new[:]=d_solv_k
        print(dy_new)
        s_SUC2[:,i] = np.transpose((dy_new[0,:,:] - SUC2) / d_Beta)
        s_Mig1[:,i] = np.transpose((dy_new[1,:,:] - Mig1) / d_Beta)
        s_Mig1_P[:, i] = np.transpose((dy_new[2, :, :] - Mig1_P) / d_Beta)
        #print(s_A, s_B)
    return SUC2, Mig1, Mig1_P

S=np.array(calc_S_mat())
print(S.shape)
def Variations_koefficient(S):
    S_T=np.transpose(S,(0,2,1))
    H=2*np.matmul(S_T,S)
    H_inv=np.linalg.inv(H)
    print(H_inv)
    C=np.diag(H_inv[1])
    for i in range(len(Kinetic_constants)):
        Var_K=C/Kinetic_constants[i]
    return Var_K

print(Variations_koefficient(Sensitivity))