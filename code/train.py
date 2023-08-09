import math
import pickle
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 28})
import numpy as np
import helper
import data_loader as dl
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, \
    Normalizer
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

memory_sizes = [128, 256, 512, 1024, 2048, 3008, 4096, 16384]
# valid_columns = ['id', 'duration', 'maxRss', 'fsRead', 'fsWrite', 'vContextSwitches', 'ivContextSwitches',
#                  'userDiff', 'sysDiff', 'rss', 'heapTotal', 'heapUsed', 'netByRx', 'netPkgRx', 'netByTx', 'netPkgTx']
used_features = [
    'userDiff_mean', 'sysDiff_mean', 'rel_vContextSwitches_mean', 'rel_userdiff_cov',
    'rel_userdiff_mean', 'rel_sysdiff_mean', 'rel_fsWrite_mean', 'rel_netByRx_mean',
    'heapUsed_mean', 'heapUsed_cov',  'mallocMem_cov']


# Train model
models = {}
scs = {}
scs2 = {}
for memory_size in memory_sizes:
    # Get dataset
    dataset = dl.loadTrainingData().reset_index(drop=True)
    dataset = dataset[dataset['f_size'] == memory_size]
    dataset = helper.ratioY(dataset)
    dataset.drop(['f_size'], axis=1, inplace=True)

    # Split dataset
    y = pd.DataFrame()
    y['y_128'] = dataset['y_128']
    y['y_256'] = dataset['y_256']
    y['y_512'] = dataset['y_512']
    y['y_1024'] = dataset['y_1024']
    y['y_2048'] = dataset['y_2048']
    y['y_3008'] = dataset['y_3008']
    y['y_4096'] = dataset['y_4096']
    y['y_16384'] = dataset['y_16384']
    y = y.drop(['y_' + str(memory_size)], axis=1)

    # Feature selection
    X = dataset.drop(dataset.columns.difference(used_features), 1)
    col_order = X.columns
    print(col_order)

    # Scaling
    sc2 = QuantileTransformer()
    X = sc2.fit_transform(X)
    sc = MinMaxScaler(clip=True)
    X = sc.fit_transform(X)

    # Build model
    model = KerasRegressor(build_fn=helper.getOptimizedModel, verbose=0)
    model.fit(X, y, epochs=helper.getOptimizedTrainingTime(), batch_size=32, verbose=1)

    models[memory_size] = model
    scs[memory_size] = sc
    scs2[memory_size] = sc2

with open('../models/models.pickle', 'wb') as handle:
    pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../models/scs.pickle', 'wb') as handle:
    pickle.dump(scs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../models/scs2.pickle', 'wb') as handle:
    pickle.dump(scs2, handle, protocol=pickle.HIGHEST_PROTOCOL)
