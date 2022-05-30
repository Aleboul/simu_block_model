import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn-whitegrid')

df_200 = pd.read_csv('results_model_4_ECO_100.csv', index_col = 0)
#df_1600 = pd.read_csv('results_model_1_1600.csv', index_col = 0)

print(df_200)
#print(df_1600)

exact_recov_rate_200 = df_200.mean(axis = 1)
#exact_recov_rate_1600= df_1600.mean(axis = 1)

n = [100,200,300,400,500,600,700,800,900,1000]

fig, ax = plt.subplots()
ax.plot(n, exact_recov_rate_200,color = 'lightblue', marker= 'd', linestyle = 'solid', markerfacecolor = 'white')
#ax.plot(n, exact_recov_rate_1600, color = 'salmon', marker= 'd', linestyle = 'solid', markerfacecolor = 'white')
ax.set_xlabel("n")
ax.set_ylabel("Exact recovery rate")
fig.savefig("exact_recov_rate.pdf")
