import numpy as np
'''
Concentration is given in fluorescence of tag on protein (Mig1, Mig1*) 
and fluorescence of tag on gene promoter (SUC2, HXK1), or unitless.
They can also be interpreted simply as signals.

Variable "concentration[n]" given as:
n  State             
0  Mig1
1  Mig1*
2  SUC2
3  X

Variable kinetic_parameter[n] are given as:
n  Parameter
0  k1
1  k2
2  k3
3  k4
4  k5
5  k6
6  k7  #testing to see if we can get mig1* to decrease faster
See the model and ODEs in "pathwaymodel_and_odes.png"
'''


def pathway_ode0(t, concentration, kinetic_parameter):
    # Function that defines the system of ODEs according to "pathwaymodel_and_odes.png"
    # "Glu" in model is lumped together with k1
    dMig_dt = concentration[1]*kinetic_parameter[0] - concentration[0]*kinetic_parameter[1] + concentration[3]*kinetic_parameter[5]
    dMigPhos_dt = -concentration[1]*kinetic_parameter[0] + concentration[0]*kinetic_parameter[1]
    dSuc2_dt = kinetic_parameter[2]/(concentration[0]+0.1) - concentration[2]*kinetic_parameter[3]
    dX_dt = kinetic_parameter[4]/(concentration[0]+0.1) - concentration[3]*kinetic_parameter[5]
    return(dMig_dt, dMigPhos_dt, dSuc2_dt, dX_dt)

