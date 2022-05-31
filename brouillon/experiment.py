import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
plt.style.use('seaborn-whitegrid')

from coppy.rng.evd import Husler_Reiss

d1 = 25

def Gamma2Sigma(Gamma, k=1, full = False):
    d = Gamma.shape[0]
    if full:
        mat = 1/2 * (np.matrix(np.repeat(Gamma[:,k], d)).reshape((d,d)) + np.matrix(np.repeat(Gamma[:,k], d)).reshape((d,d)).T - Gamma)
        return mat
    else :
        index = np.arange(d)
        mat = 1/2 * (np.matrix(np.repeat(Gamma[index != k, :][:,k], d-1)).reshape((d-1,d-1)) + np.matrix(np.repeat(Gamma[index != k,:][:,k],d-1)).reshape((d-1,d-1)).T - Gamma[index != k, :][:, index != k])
        return mat

def Sigma2Gamma(Sigma, k = 0, full = False):
    # complete S
    if not full:
        d = Sigma.shape[0] + 1
        S_full = np.insert(Sigma, 0, np.repeat(0,d-1), axis = 0)
        S_full = np.insert(S_full, 0, np.repeat(0,d), axis = 1)

        shuffle = np.arange(d)
        shuffle[shuffle <= k] = shuffle[shuffle <= k] -1
        shuffle[0] = k
        shuffle = np.sort(shuffle)

        S_full[shuffle, :][:,shuffle]
    else:
        S_full = Sigma

    One = np.ones((1,d))
    D = np.diag(S_full).reshape((1,d))
    Gamma = One.T @ D + D.T @ One - 2 * S_full

    return Gamma

def sym_matrix(d, entry = 0.5, alea = True, a = 0.0, b=1.0):
    output = np.zeros((d,d))
    entry = np.random.uniform(a, b, size = 1)
    for i in range(0,d):
        for j in range(0,i):
            if i == j:
                output[i,i] = 0.0
            else :
                if alea:
                    output[i,j], output[j,i] = entry, entry
                else :
                    output[i,j], output[j,i] = entry, entry

    return np.matrix(output)

Gamma = sym_matrix(d1, alea = True, a = 0.2, b=1.0)

copula1 = Husler_Reiss(n_sample = 512, d = d1, Sigma = Gamma)
sample = copula1.sample_unimargin()
print(sample)

fig, ax = plt.subplots()
ax.scatter(sample[:,12], sample[:,1], edgecolors= 'red', color = 'white')

plt.savefig('scatter.pdf')#