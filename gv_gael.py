import numpy as np
import pylab as pl
from scipy import stats

from sklearn import covariance
from sklearn.utils import check_random_state

import spd_manifold

######################################################################################
# Helper functions
def plot_cor(mat, vmax=None, title=None):
    "Plot correlation matrix"
    if vmax is None:
        vmax = np.max(np.abs(diff))          
    pl.matshow(mat, vmin=-vmax, vmax=vmax, cmap=pl.cm.RdBu_r)
    pl.xticks(range(len(columns)), columns)
    pl.yticks(range(len(columns)), columns)
    if title:
        pl.title(title)


def correlation(X):
    "Compute correlation matrix"
    X = X - X.mean(axis=0)
    X /= X.std(axis=0)
    cov, _ = covariance.ledoit_wolf(X) # To have robust correlations, use MCD, Minimum cov determinant, instead
    return cov


def compute_diff(X, status):
    "Compute the difference between population 1 and 2"
    X_1 = X[status == 1]
    X_2 = X[status == 2]
    cov_1 = correlation(X_1)
    cov_2 = correlation(X_2)
    # projection takes as a 1st argument the covariances to project and as a 2nd
    # argument the covariances forming the reference population
    res_1, res_2 = spd_manifold.projection([cov_1, cov_2], [cov_1, cov_2])
    diff = res_1 - res_2
    return diff


def center_permutation(center, status, random_state):
    out = status.copy()
    for c in np.unique(center):
        mask = center == c
        out[mask] = random_state.permutation(status[mask])
    return out


def permute_diff(X, center, status, random_state=0, n_perm=100):
    random_state = check_random_state(random_state)
    test_stat = []
    for i in range(n_perm):
        status0 = center_permutation(center, status, random_state)
        diff = compute_diff(X, status0)
        # 1. max T stats (multiple comparisons correction)
        #diff = np.tril(diff, k=-1)
        #test_stat.append(np.abs(diff).max())
        # 2. stats from null distribution
        test_stat.append(np.abs(diff))
    return test_stat


######################################################################################
# Data loading and massaging
columns = ['th', 'ca', 'pu', 'pa', 'hip', 'amy', 'acc', 'icv']    

data = np.recfromtxt('abide.txt', names=True)
center = data['center']
status = data['status']
X = np.c_[[data[c] for c in columns]].T

X_ASD = X[status == 1]
X_control = X[status == 2]

# Extract correlation matrices
plot_cor(correlation(X_ASD), title="ASD", vmax=1)
plot_cor(correlation(X_control), title="Control", vmax=1)

diff = compute_diff(X, status)
plot_cor(diff, title="ASD - Control")

test_stat=permute_diff(X, center, status, random_state=0, n_perm=1000)
threshold=stats.scoreatpercentile(test_stat,90) #to display p-values, use percentile at score
diff[np.abs(diff)<threshold]=0
plot_cor(diff,title="Thresholded difference")

pl.show()