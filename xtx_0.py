
#%%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


#%%
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import seaborn as sns
import time
import pickle
from main import score
sns.distplot()

#%% [markdown]
# ### data preparation

#%%
# read data
data = pd.read_csv("data-training.csv")
data_array = np.array(data)

# show sample
print(data[0:2])

# plot distribution
sns.distplot(data_array[:, 60], kde=False)


#%%
new_data = data.dropna(axis=0)
new_data = new_data.drop_duplicates()
print('data size:', new_data.shape)


#%%
# split to train & test
train_data  = new_data[0:2000000]
test_data = new_data[2000000:2700000]


#%%
diff_1st = data['askRate0'] - data['bidRate0']
diff_2nd = np.diff(diff_1st)
plt.scatter(diff_2nd, np.diff(data['y']))
plt.show()


#%%
train_x = np.array(train_data[train_data.columns[0:60]])
train_y = np.array(train_data['y'])
test_x = np.array(test_data[test_data.columns[0:60]])
test_y = np.array(test_data['y'])

#%% [markdown]
# ### score

#%%
with open('normed_data.pickle', 'rb') as f:
    normed_data = pickle.load(f)

#%% [markdown]
# ### askrate0 diff

#%%
data_diff = np.zeros((data_array.shape[0], 30), dtype=np.float32)


#%%
flag1 = np.all(data_array[:-1, 30:60] == data_array[1:, 30:60], axis=1)

flag2 = np.all(data_array[:-1, 1:15] == data_array[1:, 0:14], axis=1)
flag2_ = np.all(data_array[:-1, 16:30] == data_array[1:, 15:29], axis=1)

flag3 = np.all(data_array[:-1, 0:14] == data_array[1:, 1:15], axis=1)
flag3_ = np.all(data_array[:-1, 15:29] == data_array[1:, 16:30], axis=1)

flag = flag1 & ((flag2 & flag2_) | (flag3 & flag3_))


#%%
nb = data.shape[0] - 1
data_t1 = np.array(data.loc[np.arange(nb)[flag]+1])
data_t0 = np.array(data.loc[np.arange(nb)[flag]])
diff_ask = (data_t1[:, 0] - data_t0[:, 0]) #/ (data_t0[:, 0] - data_t0[:, 30])
diff_ask_bid = data_t0[:, 0] - data_t0[:, 30]
diff_y = data_t1[:, -1] - data_t0[:, -1]


#%%
plt.scatter(diff_ask_bid[diff_ask==0.5], diff_y[diff_ask==0.5])
plt.show()


#%%
plt.scatter(diff_ask, diff_y)
plt.show()


#%%
scipy.stats.pearsonr(diff_ask, diff_y)

#%% [markdown]
# ### data normalization

#%%
get_ipython().run_cell_magic('time', '', 'data_array = np.nan_to_num(data_array, 0)')


#%%
get_ipython().run_cell_magic('time', '', '\nnormed_data = np.zeros_like(data_array, dtype=np.float32)\nnormed_data[:, :15] = data_array[:, :15] - np.expand_dims(data_array[:, 0], axis=1)\nnormed_data[:, 30:45] = data_array[:, 30:45] - np.expand_dims(data_array[:, 0], axis=1)\n\nasksize_book = {x:0 for x in np.arange(1500, 1800, 0.5)}\nbidsize_book = {x:0 for x in np.arange(1500, 1800, 0.5)}\n\ntemp_asksize_book = {}\ntemp_bidsize_book = {}\nfor i in range(data_array.shape[0]):\n    for j in range(15):\n        # for ask\n        if data_array[i, j] not in temp_asksize_book and j < 12:\n            normed_data[i, j+15] = data_array[i ,j+15]\n        else:\n            normed_data[i, j+15] = data_array[i, j+15] - asksize_book[data_array[i, j]]\n        # update ask size book\n        asksize_book[data_array[i, j]] = data_array[i, j+15]\n    \n    for j in range(30, 45):\n        # for bid\n        if data_array[i, j] not in temp_bidsize_book and j < 42:\n            normed_data[i, j+15] = data_array[i ,j+15]\n        else:\n            normed_data[i, j+15] = data_array[i, j+15] - bidsize_book[data_array[i, j]]\n        # update bid size book\n        bidsize_book[data_array[i, j]] = data_array[i, j+15]\n        \n    # update temp book\n    temp_asksize_book = {data_array[i,k]: data_array[i,k+15] for k in range(15)}\n    temp_bidsize_book = {data_array[i,k+30]: data_array[i,k+45] for k in range(15)}\n        \n    ')


#%%
normed_data[normed_data < -1000] = 0

#%% [markdown]
# # model
#%% [markdown]
# ### linear model

#%%
nb_train = 2000000


#%%
x0 = data_array[:nb_train, 0:60]# - data_array[:nb_train, 30:60]
train_y = data_array[:nb_train, 60]

x1 = data_array[nb_train:, 0:60]# - data_array[nb_train:, 30:60]
test_y = data_array[nb_train:, 60]

reg = LinearRegression(normalize=True).fit(x0, train_y)


#%%
train_pred = reg.predict(x0)
test_pred = reg.predict(x1)


print('train score: {:.5f}'.format(score(train_y, train_pred)))
print('test score: {:.5f}'.format(score(test_y, test_pred)))

#%% [markdown]
# ### linear model with normalized data

#%%
with open('normed_data.pickle', 'rb') as f:
    normed_data = pickle.load(f)


#%%
nb_train = 2000000
# train data
train_x = normed_data[:nb_train, :60]
train_y = normed_data[:nb_train, 60]
# test data
test_x = normed_data[nb_train:, :60]
test_y = normed_data[nb_train:, 60]

reg = LinearRegression(normalize=False).fit(train_x, train_y)


#%%
train_pred = reg.predict(train_x)
test_pred = reg.predict(test_x)


print('train score: {:.5f}'.format(score(train_y, train_pred)))
print('test score: {:.5f}'.format(score(test_y, test_pred)))

#%% [markdown]
# ### linear model with multi - normalized data

#%%
def tile(x, n):
    res = np.zeros((x.shape[0] - n + 1, x.shape[1], n), dtype=np.float32)
    for i in range(n):
        if i > 0:
            res[..., i] = x[n-i-1 : -i]
        else:
            res[..., 0] = x[n-1:]
    return res.reshape((res.shape[0], -1))


#%%
n = 8
# train data
train_x = tile(normed_data[:nb_train, :60], n)
train_y = data_array[n-1:nb_train, 60]
# test data
test_x = tile(normed_data[nb_train:, :60], n)
test_y = data_array[nb_train+n-1:, 60]

reg = LinearRegression(normalize=False).fit(train_x.reshape(train_x.shape[0], -1), train_y)

# test
train_pred = reg.predict(train_x.reshape(train_x.shape[0], -1))
test_pred = reg.predict(test_x.reshape(test_x.shape[0], -1))


print('train score: {:.5f}'.format(score(train_y, train_pred)))
print('test score: {:.5f}'.format(score(test_y, test_pred)))

#%% [markdown]
# ### linear model with pca

#%%
pca = PCA(n_components=4)
pca = pca.fit(train_x)

x0 = pca.transform(train_x)
x1 = pca.transform(test_x)

reg = LinearRegression().fit(x0, train_y)


#%%
pca.explained_variance_ratio_


#%%
train_pred = reg.predict(x0)
test_pred = reg.predict(x1)

print('train score: {:.5f}'.format(score(train_y, train_pred)))
print('test score: {:.5f}'.format(score(test_y, test_pred)))


