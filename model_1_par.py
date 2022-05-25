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

def find_max(M, S):
    mask = np.zeros(M.shape, dtype = bool)
    values = np.ones((len(S),len(S)),dtype = bool)
    mask[np.ix_(S,S)] = values
    np.fill_diagonal(mask,0)
    max_value = M[mask].max()
    i , j = np.where(np.multiply(M, mask * 1) == max_value) # Sometimes doublon happens for excluded clusters, if n is low
    return i[0], j[0]


def clust(Theta, n, alpha = None):
    """ Performs clustering in AI-block model
    
    Inputs
    ------
        Theta : extremal correlation matrix
        alpha : threshold, of order sqrt(ln(d)/n)
    
    Outputs
    -------
        Partition of the set \{1,\dots, d\}
    """
    d = Theta.shape[1]

    # Initialisation

    S = np.arange(d)
    l = 0

    if alpha is None:
        alpha = 2 * np.sqrt(np.log(d)/n)
    
    cluster = {}
    while len(S) > 0:
        l = l + 1
        if len(S) == 1:
            cluster[l] = np.array(S)
        else :
            a_l, b_l = find_max(Theta, S)
            if Theta[a_l,b_l] < alpha :
                cluster[l] = np.array([a_l])
            else :
                index_a = np.where(Theta[a_l,:] >= alpha)
                index_b = np.where(Theta[b_l,:] >= alpha)
                cluster[l] = np.intersect1d(index_a,index_b)
        S = np.setdiff1d(S, cluster[l])
    
    return cluster

def perc_exact_recovery(O_hat, O_bar):
    value = 0
    for true_clust in O_bar:
        for est_clust in O_hat:
            test = np.intersect1d(true_clust,est_clust)
            if len(test) > 0 and test == true_clust :
                value +=1
    return value / len(O_hat)

def init_pool_processes():
    sp.random.seed()

def operation_model_1(dict, seed):
    """ Operation to perform Monte carlo simulation

    Input
    -----
        dict : dictionnary containing
                - d1 : dimension of the first sample
                - d2 : dimension of the second sample
                - n_sample : sample's length
    """

    sp.random.seed(1*seed)
    # Generate first sample
    copula1 = Logistic(n_sample = dict['n_sample'], d = dict['d1'], theta = 0.7)
    sample1 = copula1.sample_unimargin()
    # Generate second sample
    copula2 = Logistic(n_sample = dict['n_sample'], d = dict['d2'], theta = 0.7)
    sample2 = copula2.sample_unimargin()
    # Merge both samples to obtain one
    sample = np.hstack((sample1,sample2))
    # initialization
    d = sample.shape[1]

    R = np.zeros([dict['n_sample'], d])
    for j in range(0,d):
        X_vec = sample[:,j]
        R[:,j] = ecdf(X_vec)
    
    Theta = np.ones([d,d])
    for j in range(0,d):
        for i in range(0,j):
            Theta[i,j] = Theta[j,i] = 2 - theta(R[:,[i,j]])

    O_hat = clust(Theta, n = dict['n_sample'])
    O_bar = {1 : np.arange(0,dict['d1']), 2 : np.arange(dict['d1'],d)}

    perc = perc_exact_recovery(O_hat, O_bar)

    return perc

import multiprocessing as mp

d1 = 100
d2 = 100
n_sample = [100,200,300,400,500,600,700,800,900,1000]
n_iter = 100
pool = mp.Pool(processes= 10, initializer=init_pool_processes)

stockage = []

for n in n_sample:

    input = {'d1' : d1, 'd2': d2, 'n_sample' : n}

    result_objects = [pool.apply_async(operation_model_1, args = (input,i)) for i in range(n_iter)]

    results = [r.get() for r in result_objects]

    stockage.append(results)

    df = pd.DataFrame(stockage)

    print(df)

pool.close()
pool.join()


df.to_csv('results_model_1_200.csv')


