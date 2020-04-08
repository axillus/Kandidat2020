#Samla information till modell samt lösning

import numpy as np
import scipy.integrate as integrate
from Main import Kinetic_constants
from Main import y0
from Main import t_span
from Main import t_eval
from Main import Num_eq
from Main import Num_timestep
from Main import model1

def sol():
    solv=integrate.solve_ivp(fun=lambda t, y: model1(t, y, Kinetic_constants), t_span=t_span, y0=y0, method="RK45",
                             t_eval=t_eval)
    solv_k = np.zeros([Num_eq, Num_timestep, 1])
    solv_k[:, :, 0] = solv.y
    return solv_k

Mig1, Mig1P, SUC2, X=sol()
