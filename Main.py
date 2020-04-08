#Samla allmän information här, startvärden, parametervärden, tid, modelstruktur osv.
import numpy as np

Kinetic_constants = [7.62822144e+01, 1.30227426e-02, 5.56075536e-01, 3.12094030e-02, 1.52017194e+02, 1.17756944e+02,
                     2.63970429e+01]
t_span = [0, 300]
t_eval=np.linspace(0,280,num=100)
y0 = [2.64986, 0.001, 4.04353, 0.001]
Num_eq=4
Num_timestep=len(t_eval)

def model1(t,y,Kinetic_constants):
    Mig1 = y[0]
    Mig1P = y[1]
    SUC2 = y[2]
    X = y[3]
    r1 = Kinetic_constants[0] * Mig1P
    r2 = Kinetic_constants[1] * Mig1
    r3 = Kinetic_constants[2] / (Mig1+0.01)
    r4 = Kinetic_constants[3] * SUC2
    r5 = Kinetic_constants[4] / (Mig1+0.01)
    r6 = Kinetic_constants[5] * X
    r7 = Kinetic_constants[6] * Mig1
    dMig1_dt = r1 - r2 + r6 - r7
    dMig1P_dt = - r1 + r2
    dSUC2_dt= r3 - r4
    dX_dt= r5 - r6
    dy_dt = [dMig1_dt, dMig1P_dt, dSUC2_dt, dX_dt]
    return dy_dt