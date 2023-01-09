## uisng the YAML file generated we need to build the model using sequential()

import tensorflow as tf
import numpy as np
import sklearn
import random,os,io,pandas as pd, yaml,sys
from tensorflow import keras
from tensorflow.keras import layers
import yaml
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.callbacks import Callback, EarlyStopping, LearningRateScheduler, ModelCheckpoint


def create_model(params):
    model = keras.Sequential()
    model.add(layers.Input(16))
    num_of_layers=int(list(params.keys())[-2][-1])
    print(num_of_layers)
    for i in range(num_of_layers):
        print(i)
        model.add(layers.Dense(units=params[f'units_{str(i+1)}'],kernel_initializer=params['weights'],
                               activation=params[f'activation_{str(i+1)}']))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            params['learning_rate']),
        loss='mean_absolute_error',
        metrics=["mean_squared_error","mean_absolute_error","mean_absolute_percentage_error","mean_squared_logarithmic_error"])
    return model

if __name__=="__main__":
    dep_train=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\XTr.dat",sep='\s+',names=[str(i) for i in range(0,16)])
    indep_train=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\yTr.dat",sep='\s+',names=['target'])
    dep_val=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\XV.dat",sep='\s+',names=[str(i) for i in range(0,16)])
    indep_val=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\yV.dat",sep='\s+',names=["target"])

    with open("params1.yaml", 'r') as stream:
        params = yaml.safe_load(stream)
    model=create_model(params=params)

    history=model.fit(dep_train, indep_train, batch_size=32, epochs=3, validation_data=(dep_val, indep_val))
    model.save("model")
    pd.DataFrame(history.history).to_csv("metrices.csv",index=False)



