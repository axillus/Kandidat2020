#Heat maps för Korrelationerna

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from Numerisk_ID import Correlation

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

x_axis_labels=['K1', 'K2', 'K3', 'K4', 'K5', 'K6']
y_axis_labels=['K1', 'K2', 'K3', 'K4', 'K5', 'K6']

sns.set(style="white")
mask = np.triu(np.ones_like(Correlation[0,:,:], dtype=np.bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

Corr_Mig1=sns.heatmap(Correlation[0,:,:], mask=mask, cmap=cmap, vmin=-1, vmax=1, xticklabels=x_axis_labels, yticklabels=y_axis_labels
                      , square=True, linewidths=.5, cbar_kws={"shrink": .8}).set_title('Korrelation Mig1', fontdict=font)
plt.show()

Corr_SUC2=sns.heatmap(Correlation[2,:,:], mask=mask, cmap=cmap, vmin=-1, vmax=1, xticklabels=x_axis_labels, yticklabels=y_axis_labels
                      , square=True, linewidths=.5, cbar_kws={"shrink": .8}).set_title('Korrelation SUC2', fontdict=font)
plt.show()