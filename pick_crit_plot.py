import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-whitegrid')

df = pd.read_csv('SECO_25_75_HR.csv')
df.drop(columns=df.columns[0], axis=1, inplace=True)
df = df.stack()
print(df[0].values)

df_plot = pd.DataFrame()
df_plot['values'] = df[0].diff()
df_plot['d'] = np.arange(len(df_plot))+1

for i in range(1,200):
    _df = pd.DataFrame()
    _df['values'] = -df[i].diff()
    _df['d'] = np.arange(len(_df))+1
    df_plot = pd.concat([df_plot, _df])


fig, ax = plt.subplots()
#ax.plot(d_, median, linewidth = 1, c = 'orange')
#ax.boxplot(np.array(DF), positions = d_, showfliers= False)
ax = sns.boxplot(x = "d", y = "values", data = df_plot, showfliers= False, palette = 'crest', linewidth = 0.5)
ax.set_xlabel('d')
ax.set_ylabel(r'$\delta_n^{\mathcal{H}}$')

plt.savefig("boxplot.pdf")

