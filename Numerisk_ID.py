#Tar fram Kovariansmatrisen, variationskoefficienter samt Korrelationsmatrisen

import numpy as np
from Main import Kinetic_constants
from Main import Num_eq
from Sensitivity import S
import seaborn as sns

def Var(S):
    Var_K = np.zeros((Num_eq, len(Kinetic_constants),1))
    S_T=np.transpose(S,(0,2,1))
    H=2*np.matmul(S_T,S)
    H_inv=np.linalg.inv(H)
    for i in range(Num_eq):
        for j in range(len(Kinetic_constants)):
            for k in range(len(Kinetic_constants)):
                C=np.array(np.diag(H_inv[i,:,:]))
                Var_K[Kinetic_constants == 0] = 0
                Var_K[i,k,:]=C[k]/Kinetic_constants[j]
    return H_inv, Var_K

Covariance, Var_K = Var(S)


def Corr(Covariance):
    v0 = np.zeros((len(Kinetic_constants),1))
    Correlation = np.zeros((2,len(Kinetic_constants),len(Kinetic_constants)))
    for i in range(Num_eq):
        v = v0.copy()
        v = np.sqrt(np.diag(Covariance[i,:,:]))
        outer_v = np.outer(v, v)
        Correlation[Covariance == 0] = 0
        Correlation[i,:,:] = Covariance[i,:,:] / outer_v
    return Correlation

Correlation=Corr(Covariance)

x_axis_labels=['K1', 'K2', 'K3']
y_axis_labels=['K1', 'K2', 'K3']

plt.subplot(221)
Corr_Mig1=sns.heatmap(Correlation[0,:,:], vmin=-1, vmax=1, xticklabels=False, yticklabels=y_axis_labels).set_title('Korrelation')


plt.subplot(223)
Corr_SUC2=sns.heatmap(Correlation[2,:,:], vmin=-1, vmax=1, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.show()
