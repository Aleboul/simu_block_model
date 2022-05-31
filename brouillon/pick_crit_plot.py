import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

d = 100
d1 = 25
plt.style.use('seaborn-whitegrid')

df = pd.read_csv('CRIT_25_75_HR.csv')
df.drop(columns=df.columns[0], axis=1, inplace=True)
df = df.stack()
df = df * d
df_plot = pd.DataFrame()
df_plot['SECO'] = -df[0][range(d1)].diff()
df_plot['d'] = np.arange(0,d1)

for i in range(1,200):
    _df = pd.DataFrame()
    _df['SECO'] = -df[i][range(d1)].diff()
    _df['d'] = np.arange(0,d1)
    df_plot = pd.concat([df_plot, _df])

print(df_plot.d.values)

fig, ax = plt.subplots()
#ax.plot(d_, median, linewidth = 1, c = 'orange')
#ax.boxplot(np.array(DF), positions = d_, showfliers= False)
ax = sns.boxplot(x = "d", y = "SECO", data = df_plot, showfliers= False, palette = 'crest', linewidth = 0.5)
ax.set_xlabel('d')
ax.set_ylabel(r'$\delta_n^{\mathcal{H}}$')

plt.savefig("boxplot_dep.pdf")

#fig, ax = plt.subplots()
#ax.plot(np.arange(d), df[0])
#plt.savefig('pick_crit.pdf')

df_plot = pd.DataFrame()
df_plot['SECO'] = df[0][d1-1] - df[0][(d1):].values
df_plot['d'] = np.arange(d1,d)

for i in range(1,200):
    _df = pd.DataFrame()
    _df['SECO'] = df[i][d1-1] - df[i][(d1):].values
    _df['d'] = np.arange(d1,d)
    df_plot = pd.concat([df_plot, _df])

print(df_plot)

fig, ax = plt.subplots()
#ax.plot(d_, median, linewidth = 1, c = 'orange')
#ax.boxplot(np.array(DF), positions = d_, showfliers= False)
ax = sns.boxplot(x = "d", y = "SECO", data = df_plot, showfliers= False, palette = 'crest', linewidth = 0.5)
ax.set_xlabel('d')
ax.set_ylabel(r'$\delta_n^{\mathcal{H}}$')

plt.savefig("boxplot_ind.pdf")
