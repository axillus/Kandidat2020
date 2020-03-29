import numpy as np
import math
import scipy.integrate as integrate

Kinetic_constants = [k1, k2, k2...]
t_span = [0 100]
t_eval=np.linspace(0, 100, num=10)
y0 = [y01, y02]
eps = np.finfo(float).eps
d_Beta = math.sqrt(eps)
Num_eq=#Ekvationer
Num_timestep=len(t_eval)

def simp_model(t,y,Kinetic_constants):
    Mig1=y[0]
    Mig1_P=y[1]
    SUC2 = y[2]
    X=y[3]
    r1=Kinetic_constants[0]*A...
    r2=Kinetic_constants[1]*B...
    r3=...
    dSUC2_dt=-r1+r2
    dMig1_dt=-r2+r1
    dy_dt=[dSUC2_dt, dMig1_dt]
    return dy_dt


def sol():
    solv=integrate.solve_ivp(fun=lambda t, y: simp_model(t, y, Kinetic_constants), t_span=t_span, y0=y0, method="RK45",
                             t_eval=t_eval)
    solv_k = np.zeros([Num_eq, Num_timestep, 1])
    solv_k[:, :, 0] = solv.y
    return solv_k

SUC2, Mig1, Mig1P, X=sol()


def calc_S_mat(Kinetic_constants):
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
        s_Mig1[:,i] = np.transpose((dy_new[0,:,:] - Mig1) / d_Beta)
        s_Mig1_P[:,i] = np.transpose((dy_new[1,:,:] - Mig1_P) / d_Beta)
        s_SUC2[:, i] = np.transpose((dy_new[2, :, :] - SUC2) / d_Beta)
        s_X[:,i] = np.transpose((dy_new[3, :, :] - X) / d_Beta)
    return Mig1, Mig1_P, SUC2, X

S=np.array(calc_S_mat(Kinetic_constants))
def Var_koefficient(S):
    S_T=np.transpose(S,(0,2,1))
    H=2*np.matmul(S_T,S)
    H_inv=np.linalg.inv(H)
    C=np.diag(H_inv[1])
    for i in range(len(Kinetic_constants)):
        Var_K=C/Kinetic_constants[i]
    return Var_K

print(Var_koefficient(S))