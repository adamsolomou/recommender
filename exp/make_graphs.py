import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%

data = pd.read_csv('autoencoder_exp_depth.csv')
data['layers'] = data.Architecture.apply(eval)
data['size'] = data['layers'].apply(len)
data['layer'] = data['layers'].apply(lambda x: x[0])

len_1 = data[data['size'] == 1]
len_2 = data[data['size'] == 2]
len_3 = data[data['size'] == 3]

mark_1 = [np.argmin(len_1['Score'].values)]
mark_2 = [np.argmin(len_2['Score'].values)]
mark_3 = [np.argmin(len_3['Score'].values)]


plt.plot(len_1['layer'], len_1['Score'], '-o', markevery=mark_1)
plt.plot(len_2['layer'], len_2['Score'], '-o', markevery=mark_2)
plt.plot(len_3['layer'], len_3['Score'], '-o', markevery=mark_3)

plt.xticks(range(2,21))

plt.xlabel('Neurons per layer')
plt.ylabel('RMSE')

plt.legend(['1-layer', '2-layers', '3-layers'], loc='upper left')
plt.savefig('autoenc_layers.pdf')

#%%
data = pd.read_csv('autoencoder_exp_masking.csv')
data['masking'] = data['masking'].apply(float)

plt.xlabel('Dropout')
plt.ylabel('RMSE')

plt.plot(data['masking'], data['Score'], '-o', markevery=[np.argmin(data['Score'].values)])
plt.savefig('autoenc_masking.pdf')