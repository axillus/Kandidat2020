# Gives four plots that shows in which intervals the different parameters give rise to an oscillating modell
# Blue = Stiff model
# White = Oscillating model

import seaborn as sns
import matplotlib.pylab as plt


from Binary_Eigenvalues import f1, f2, f3, f4, f5

fig,axn = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6, 6))

ax = plt.subplot(2, 2, 1)
sns.heatmap(
    f1, cmap='RdBu_r', xticklabels=40, yticklabels=40, cbar=False, ax=ax)
plt.xlabel("k1")
plt.ylabel("k2")
ax.set_aspect('equal')

ax = plt.subplot(2, 2, 2)
sns.heatmap(
    f2, cmap='RdBu_r',xticklabels=40, yticklabels=40, cbar=False, ax=ax)
plt.xlabel("k4")
plt.ylabel("k5")
ax.set_aspect('equal')

ax = plt.subplot(2, 2, 3)
sns.heatmap(
    f3, cmap='RdBu_r', xticklabels=40, yticklabels=40, cbar=False, ax=ax)
plt.xlabel("k5") # f3 -> k5 & k7, f4 -> k1 & k6
plt.ylabel("k7")
ax.set_aspect('equal')

ax = plt.subplot(2, 2, 4)
sns.heatmap(
    f5, cmap='RdBu_r', xticklabels=40, yticklabels=40, cbar=False, ax=ax)
plt.xlabel("k6") #f4 -> k1 & k6, f5 -> k6 & k7
plt.ylabel("k7")
ax.set_aspect('equal')

fig.tight_layout(rect=[1, 0, .9, 1])
fig.suptitle('Parameterintervall som ger styv eller oscillerande modell')
plt.show()