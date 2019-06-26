#%%
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Input, GaussianNoise, Lambda, Dot, BatchNormalization
from tensorflow.keras.layers import Dropout, Embedding, Concatenate, Flatten, Reshape, Add
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import MSE
#from tensorflow.keras.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.backend as KB
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# https://nipunbatra.github.io/blog/2017/neural-collaborative-filtering.html

DATA_PREPROCESS = "data/preprocessed/data_df.csv"

#%%

data = pd.read_csv(DATA_PREPROCESS)

#%% 
#data['ratings'] = ((data['Prediction'] - 3)/2).values.reshape(-1,1)



probs = np.zeros([6])
pred, counts = np.unique(data['Prediction'], return_counts=True)
sum_counts = np.sum(counts)
probs[pred] = counts / sum_counts



#%%
data['weight'] = data['Prediction'].apply(lambda x: probs[x])

#%%
#scaler = MinMaxScaler()
#scaler.fit(data['ratings'].values.reshape(-1,1))
#data['ratings'] = scaler.transform(data['ratings'].values.reshape(-1,1))

#%%

input_user = Input((1,), dtype='int32')
input_movie = Input((1,), dtype='int32')

mf_latent = 3 # Best 3
movie_mlp = 16 # 12 
user_mlp = 12 # 8
l2 = 0.0001

user_emb_mlp= Flatten()(Embedding(10001,  user_mlp)(input_user))
movie_emb_mlp = Flatten()(Embedding(1001, movie_mlp)(input_movie))

user_emb_mlp = Dropout(0.2)(user_emb_mlp)
movie_emb_mlp = Dropout(0.2)(movie_emb_mlp)

concat_mlp = Concatenate()([user_emb_mlp, movie_emb_mlp])

concat_mlp = Dropout(0.2)(concat_mlp)
dense_1 = Dense(200)(concat_mlp)
dense_1 = BatchNormalization()(dense_1)
dense_1 = Dropout(0.2)(dense_1)
dense_2 = Dense(100)(dense_1)
dense_2 = BatchNormalization()(dense_2)
dense_2 = Dropout(0.2)(dense_2)
dense_3 = Dense(50)(dense_2)
dense_4 = Dense(20, activation='selu')(dense_3)
pred_mlp = Dense(1, activation='selu')(dense_4)


user_mf = Flatten()(Embedding(10001, mf_latent)(input_user))
movie_mf = Flatten()(Embedding(1001, mf_latent)(input_movie))

user_mf = Dropout(0.2)(user_mf)
movie_mf = Dropout(0.2)(movie_mf)

pred_mf = Dot(1)([user_mf, movie_mf])

mf_mlp = Concatenate()([pred_mlp, pred_mf])
dense_ml_mlp = Dense(100)(mf_mlp)
deep_dense = Dense(100)(dense_ml_mlp)
output = Dense(1)(deep_dense)

# output = Reshape((-1,))(output)

optimizer = Adam(lr=0.001)
#optimizer = SGD()

def rmse(y_true, y_pred):
    return KB.sqrt(KB.mean(KB.square(y_pred - y_true))) 


model = Model([input_user, input_movie], output)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[rmse])

epoch = 1

#%%

user_train, user_test, movie_train, movie_test,\
rating_train, rating_test, weight_train, weight_test =\
    train_test_split(data['user'], data['movie'], data['Prediction'], data['weight'], test_size=0.12)

#%%

min_score = 0.9853

while(True):
    model.fit([user_train.values, movie_train.values], rating_train.values,
              shuffle=True, batch_size=512, epochs=1, verbose=1)
    rating_pred = model.predict([user_test, movie_test])
    score = sqrt(mean_squared_error(rating_test, rating_pred))
    print("{}:".format(epoch),score)
    if score <= min_score:
        min_score = score
        model.save("biased_neural_collaborative.h5")
    epoch += 1

#%%
