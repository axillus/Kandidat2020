# Heat maps för Korrelationerna

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from Numerisk_ID import corr


def plot_corr():
    plt.clf()
    Correlation = corr()
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 16,
            }
    x_axis_labels = ['$\hat{\u03B8}_1$', '$\hat{\u03B8}_2$', '$\hat{\u03B8}_3$', '$\hat{\u03B8}_4$',
                    '$\hat{\u03B8}_6$', '$\hat{\u03B8}_7$']
    y_axis_labels = ['$\hat{\u03B8}_1$', '$\hat{\u03B8}_2$', '$\hat{\u03B8}_3$', '$\hat{\u03B8}_4$',
                    '$\hat{\u03B8}_6$', '$\hat{\u03B8}_7$']

    sns.set(style="white")
    mask1 = np.triu(np.ones_like(Correlation[0, :, :], dtype=np.bool))
    mask2 = np.triu(np.ones_like(Correlation[1, :, :], dtype=np.bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    Corr_Mig1 = sns.heatmap(Correlation[1, :, :], mask=mask2, cmap=cmap, vmin=-1, vmax=1, xticklabels=x_axis_labels,
                        yticklabels=y_axis_labels, square=True, linewidths=.8,
                        cbar_kws={"shrink": .8}).set_title('Korrelation Mig1', fontdict=font)
    plt.show()
    Corr_SUC2 = sns.heatmap(Correlation[0, :, :], mask=mask1, cmap=cmap, vmin=-1, vmax=1, xticklabels=x_axis_labels,
                        yticklabels=y_axis_labels, square=True, linewidths=.8,
                        cbar_kws={"shrink": .8}).set_title('Korrelation SUC2', fontdict=font)
    plt.show()

plot_corr()
