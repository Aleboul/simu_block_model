import sys
from time import process_time

sys.path.insert(0, '/home/aboulin/')

from CoPY.copy.rng import evd, monte_carlo, archimedean
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, expon
plt.style.use('seaborn-whitegrid')

def gauss_function(x, x0, sigma):
    return np.sqrt(1 / (2*np.pi * sigma**2)) * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2) )

seed = 42
n_sample = 2048
theta, psi1, psi2, d = 10, 0.5, 1.0, 2

copula = evd.Asy_neg_log(theta = theta, psi1 = psi1, psi2 = psi2, n_sample = n_sample, d = d)
sample = copula.sample(inv_cdf = [norm.ppf, expon.ppf])

seagreen = sns.light_palette("seagreen", as_cmap = True)
fig, ax = plt.subplots()
ax.scatter(sample[:,0], sample[:,1], edgecolors = seagreen(0.25), color = 'white', s = 5)
ax.set_xlabel(r'$x_0$')
ax.set_ylabel(r'$x_1$')
plt.savefig('/home/aboulin/CoPY/script/output/bv_plot.pdf')

copula_miss = archimedean.Joe(theta = 2.0, n_sample = n_sample)

u = np.array([0.9,0.9])
n_iter, P, w = 1024,[[u[0], copula_miss._C(u)], [copula_miss._C(u), u[1]]], np.array([0.5,0.5])
monte = monte_carlo.Monte_Carlo(n_iter = n_iter, n_sample = n_sample, copula = copula, copula_miss = copula_miss, w = w, P = P, random_seed = seed)
df_wmado = monte.finite_sample(inv_cdf = [norm.ppf, expon.ppf], corr = True)
print(df_wmado.head())

var_mado = copula.var_mado(w, p = copula_miss._C(u), P = P, corr = True)
print(var_mado)
print(df_wmado['scaled'].var())

fig, ax = plt.subplots()
sigma = np.sqrt(var_mado)
x = np.linspace(min(df_wmado['scaled']), max(df_wmado['scaled']), 1000)
gauss = gauss_function(x, 0, sigma)
sns.displot(data = df_wmado, x = "scaled", color = seagreen(0.5), kind = 'hist',stat = 'density', common_norm = False, alpha = 0.5, fill = True, linewidth = 1.5)
plt.plot(x,gauss, color = 'darkblue')
plt.savefig('/home/aboulin/CoPY/script/output/normality.pdf')