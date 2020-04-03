#Heat maps för Korrelationerna

import seaborn as sns
import matplotlib.pyplot as plt
from Numerisk_ID import Correlation

x_axis_labels=['K1', 'K2', 'K3', 'K4', 'K5', 'K6']
y_axis_labels=['K1', 'K2', 'K3', 'K4', 'K5', 'K6']

sns.set(style="white")
mask = np.triu(np.ones_like(Correlation[0,:,:], dtype=np.bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

Corr_Mig1=sns.heatmap(Correlation[0,:,:], mask=mask, cmap=cmap, vmin=-1, vmax=1, xticklabels=x_axis_labels, yticklabels=y_axis_labels
                      , square=True, linewidths=.5, cbar_kws={"shrink": .5}).set_title('Korrelation Mig1')
plt.show(Corr_Mig1)

Corr_SUC2=sns.heatmap(Correlation[2,:,:], mask=mask, cmap=cmap, vmin=-1, vmax=1, xticklabels=x_axis_labels, yticklabels=y_axis_labels
                      , square=True, linewidths=.5, cbar_kws={"shrink": .5}).set_title('Korrelation SUC2')
plt.show(Corr_SUC2)