import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
plt.style.use('seaborn-whitegrid')

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

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
                cluster[l] = np.intersect1d(S,index_a,index_b)
        S = np.setdiff1d(S, cluster[l])
    
    return cluster


def perc_exact_recovery(O_hat, O_bar):
    value = 0
    for key1, true_clust in O_bar.items():
        for key2, est_clust in O_hat.items():
            if len(true_clust) == len(est_clust):
                test = np.intersect1d(true_clust,est_clust)
                if len(test) > 0 and np.sum(np.sort(test) - np.sort(true_clust)) == 0 :
                    value +=1
    return value / len(O_bar)

def make_sample(d, n_sample, K):
    probs = []
    for k in range(1,K-5):
        p = 1 / 2**k
        probs.append(p)

    p = 1 - np.sum(probs)
    probs.append(p)
    sizes = np.random.multinomial(d-5, probs)
    sizes = np.hstack([sizes, np.ones(5)]).astype(int)
    sample = []
    _d = 0
    O_bar = {}
    l=0
    for d_ in sizes :
        l += 1
        theta = np.random.uniform(0.65,0.75, size = 1)
        copula = Logistic(theta = theta, n_sample = n_sample, d = d_)
        sample_ = copula.sample_unimargin()
        sample.append(sample_)
        O_bar[l] = np.arange(_d, _d + d_)
        _d += d_

    sample = np.hstack(sample)

    return sample, O_bar

def init_pool_processes():
    sp.random.seed()

def operation_model_1_ECO(dict, seed):
    """ Operation to perform Monte carlo simulation

    Input
    -----
        dict : dictionnary containing
                - d1 : dimension of the first sample
                - d2 : dimension of the second sample
                - n_sample : sample's length
    """
    sp.random.seed(1*seed)
    sample, O_bar = make_sample(dict['d'], dict['n_sample'], dict['K'])

    R = np.zeros([dict['n_sample'], d])
    for j in range(0,d):
        X_vec = sample[:,j]
        R[:,j] = ecdf(X_vec)
    
    Theta = np.ones([d,d])
    for j in range(0,d):
        for i in range(0,j):
            Theta[i,j] = Theta[j,i] = 2 - theta(R[:,[i,j]])

    O_hat = clust(Theta, n = dict['n_sample'])

    perc = perc_exact_recovery(O_hat, O_bar)

    return perc

def operation_model_1_HC(dict, seed):
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

    HC = AgglomerativeClustering(n_clusters=2, affinity = 'precomputed', linkage = 'complete').fit(Theta)
    labels = HC.labels_
    label = np.unique(labels)
    O_hat = {}

    for lab, l in enumerate(label):
        l += 1
        index = np.where(labels == lab)
        O_hat[l] = index

    perc = perc_exact_recovery(O_hat, O_bar)

    return perc

def operation_model_1_KMeans(dict, seed):
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

    kmeans = KMeans(n_clusters=2).fit(sample)
    labels = kmeans.labels_
    label = np.unique(labels)
    O_hat = {}

    for lab, l in enumerate(label):
        l += 1
        index = np.where(labels == lab)
        O_hat[l] = index

    O_bar = {1 : np.arange(0,d1), 2 : np.arange(d1,d1+d2)}

    perc = perc_exact_recovery(O_hat, O_bar)

    return perc

import multiprocessing as mp

K = 10
d = 1600
n_sample = [100,200,300,400,500,600,700,800,900,1000]
n_iter = 100
pool = mp.Pool(processes= 10, initializer=init_pool_processes)
mode = "ECO"
stockage = []

for n in n_sample:

    input = {'d' : d, 'n_sample' : n, 'K' : K}

    if mode == "ECO":

        result_objects = [pool.apply_async(operation_model_1_ECO, args = (input,i)) for i in range(n_iter)]
    
    if mode == "HC":
        
        result_objects = [pool.apply_async(operation_model_1_HC, args = (input,i)) for i in range(n_iter)]
    
    if mode == "KMeans":

        result_objects = [pool.apply_async(operation_model_1_KMeans, args = (input,i)) for i in range(n_iter)]


    results = [r.get() for r in result_objects]

    stockage.append(results)

    df = pd.DataFrame(stockage)

    print(df)

pool.close()
pool.join()


df.to_csv('results_model_3_ECO_1600.csv')


