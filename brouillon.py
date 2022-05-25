import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
plt.style.use('seaborn-whitegrid')

from coppy.rng.evd import Logistic

def ecdf(X):
    """ Compute uniform ECDF.
    
    Inputs
    ------
        X (np.array[float]) : array of observations

    Output
    ------
        Empirical uniform margin
    """

    index = np.argsort(X)
    ecdf = np.zeros(len(index))

    for i in index :
        ecdf[i] = (1.0 / X.shape[0]) * np.sum(X <= X[i])

    return ecdf

def theta(R) :
    """
        This function computes the w-madogram

        Inputs
        ------
        R (array([float]) of n_sample \times d) : rank's matrix
                                              w : element of the simplex
                            miss (array([int])) : list of observed data
                           corr (True or False) : If true, return corrected version of w-madogram
        
        Outputs
        -------
        w-madogram
    """

    Nnb = R.shape[1]
    Tnb = R.shape[0]
    V = np.zeros([Tnb, Nnb])
    for j in range(0, Nnb):
        V[:,j] = R[:,j]
    value_1 = np.amax(V,1)
    value_2 = (1/Nnb) * np.sum(V, 1)
    mado = (1/Tnb) * np.sum(value_1 - value_2)

    value = (mado + 1/2) / (1/2-mado)
    return value

def find_max(M):
    mask = np.ones(M.shape, dtype = bool)
    np.fill_diagonal(mask,0)
    max_value = M[mask].max()
    i , j = np.where(M == max_value)
    return i[0], j[0]

d1 = 20
d2 = 20
n_sample = 1000

copula1 = Logistic(n_sample = n_sample, d = d1, theta = 0.5)
copula2 = Logistic(n_sample = n_sample, d = d2, theta = 0.5)
sample1 = copula1.sample_unimargin()
sample2 = copula2.sample_unimargin()
sample = np.hstack([sample1, sample2])
d = sample.shape[1]
R = np.zeros([n_sample,d])
for j in range(0,d):
    X_vec = sample[:,j]
    R[:,j] = ecdf(X_vec)

Theta = np.ones([d,d])
for j in range(0,d):
    for i in range(0,j):
        Theta[i,j] = Theta[j,i] = 2 - theta(R[:,[i,j]])

alpha = 2 * np.sqrt(np.log(d)/n_sample)
print(alpha)

S = np.arange(d)
Theta_sub = Theta[S,:][:,S]
#print(Theta_sub)
a_l, b_l = find_max(Theta_sub)
print(a_l, b_l)
print(Theta_sub[a_l,b_l])
index_a = np.where(Theta_sub[a_l,:] >= alpha)
index_b = np.where(Theta_sub[a_l,:] >=alpha)
cluster = {}
cluster[0] = np.intersect1d(index_a,index_b)

print(cluster[0])

S = np.delete(S, cluster[0])
print(S)