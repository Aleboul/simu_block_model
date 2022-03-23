import numpy as np
import time
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

import sys
sys.path.insert(0, '/home/aboulin/')

from CoPY.src.rng.utils import simplex
from CoPY.src.rng.evd import Logistic
from CoPY.src.rng.monte_carlo import Monte_Carlo

def max(a,b):
    if a >= b:
        return a
    else:
        return b

d_ = [35,40]
theta = 0.5
n_sample = [256]
n_iter = 300
sup_ = []
t_ = []
w_ = []
array_ = []
for d in d_:
    print(d)
    P = np.ones([d, d])
    p = 1.0
    copula = Logistic(theta= theta, d = d, n_sample = np.max(n_sample))
    t = time.process_time()
    for k in tqdm(range(0,300)):
        w = simplex(d = d)[0]
        Monte = Monte_Carlo(n_iter = n_iter, n_sample = n_sample, w = w, copula = copula, P = P)
        df_wmado = Monte.finite_sample([norm.ppf], corr = True)
        sigma_empi_ = df_wmado['scaled'].var()
        sigma_theo_ = max(0,copula.var_mado(w,P, p, corr = True))
        value_ = np.abs(sigma_empi_ - sigma_theo_) / sigma_theo_
        array_.append(value_)
        w_.append(w)
    t = time.process_time() - t
    t_.append(t)

df = pd.DataFrame()
df['w'] = list(w_)
df['w_mado'] = array_
df.columns = ['w', 'w_mado']
print(df)
df.to_csv("sup_35_40_256.csv")
#df = pd.read_csv("/home/aboulin/Documents/stage/papier/code/multivariate_ed/output/inc_dim/sup_2_50.csv", index_col = 0)
#print(df)
#DF = pd.DataFrame()
#print(df[df['d'] == 2])
#
## reshape data
#for d in d_:
#    DF[d] = np.array(df[df['d'] == d]['w_mado'])
#
#print(np.array(DF))
#
#median = DF.median()
#fig, ax = plt.subplots()
#ax.plot(d_, median, linewidth = 1, c = 'orange')
#plt.boxplot(np.array(DF), positions = d_, showfliers= False)
#ax.set_xlabel('d')
#ax.set_ylabel(r'$\delta_n^{\mathcal{H}}$')
#plt.savefig("/home/aboulin/Documents/stage/papier/code/multivariate_ed/output/inc_dim/boxplot.pdf")
#