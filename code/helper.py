0# Ensure reproducability
seed_value= 111
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ['SKLEARN_SEED']=str(seed_value)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.random.set_seed(seed_value)

# Imports
import pandas as pd


def getModel(optimizer='adam', loss="MSE", neurons=128, layers=3, l2=0):
  os.environ['PYTHONHASHSEED']=str(seed_value)
  os.environ['SKLEARN_SEED']=str(seed_value)
  random.seed(seed_value)
  np.random.seed(seed_value)
  tf.random.set_seed(seed_value)
  lay = []
  for l in range(0, layers-1):
    lay.append(tf.keras.layers.Dense(neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)))
  lay.append(tf.keras.layers.Dense(7))
  model = tf.keras.models.Sequential(lay)
  model.compile(optimizer=optimizer, loss=loss)
  return model

def getOptimizedTrainingTime():
  return 200

def getOptimizedModel(optimizer='Adam', loss="mean_absolute_percentage_error", neurons=256, layers=4, l2=0.01):
  os.environ['PYTHONHASHSEED']=str(seed_value)
  os.environ['SKLEARN_SEED']=str(seed_value)
  random.seed(seed_value)
  np.random.seed(seed_value)
  tf.random.set_seed(seed_value)
  lay = []
  for l in range(0,layers-1):
    lay.append(tf.keras.layers.Dense(neurons, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)))
  lay.append(tf.keras.layers.Dense(7))
  model = tf.keras.models.Sequential(lay)
  model.compile(optimizer=optimizer, loss=loss)
  return model


def ratioY(dataset):
    dataset['y_128'] = dataset['y_128'] / dataset['duration_mean']
    dataset['y_256'] = dataset['y_256'] / dataset['duration_mean']
    dataset['y_512'] = dataset['y_512'] / dataset['duration_mean']
    dataset['y_1024'] = dataset['y_1024'] / dataset['duration_mean']
    dataset['y_2048'] = dataset['y_2048'] / dataset['duration_mean']
    dataset['y_3008'] = dataset['y_3008'] / dataset['duration_mean']
    dataset['y_4096'] = dataset['y_4096'] / dataset['duration_mean']
    dataset['y_16384'] = dataset['y_16384'] / dataset['duration_mean']
    return dataset
    
def makeRelativeFeatures(X):
  X['rel_maxRss_mean'] = X['maxRss_mean']/533725184
  X['rel_vContextSwitches_mean'] = X['vContextSwitches_mean']/X['duration_mean']
  X['rel_ivContextSwitches_mean'] = X['ivContextSwitches_mean']/X['duration_mean']
  X['sysuserDiff_mean'] = X['userDiff_mean'] + X['sysDiff_mean']
  X['rel_userdiff_mean'] = X.apply(lambda row: row['userDiff_mean'] / 1000 / row['duration_mean'], axis=1)
  X['rel_sysdiff_mean'] = X.apply(lambda row: row['sysDiff_mean'] / 1000 / row['duration_mean'], axis=1)
  X['rel_sysuserdiff_mean'] = X.apply(lambda row: row['sysuserDiff_mean'] / 1000 / row['duration_mean'], axis=1)
  X['rel_heapUsed_mean'] = X['heapUsed_mean']/533725184
  X['rel_elMin_mean'] = X['elMin_mean']/X['duration_mean']
  X['rel_elMax_mean'] = X['elMax_mean']/X['duration_mean']
  X['rel_elMean_mean'] = X['elMean_mean']/X['duration_mean']
  X['rel_bytecodeMetadataSize_mean'] = X['bytecodeMetadataSize_mean']/533725184
  return X

def addDistributionFeatures(X, memory_size=512):
  dist_descriptors = pd.read_csv("../data/distribution_descriptors_" + str(memory_size) + ".csv").reset_index(drop=True)
  X = X.merge(dist_descriptors, left_on='function', right_on='function')
  X.drop(['function'], axis=1, inplace=True)
  return X
