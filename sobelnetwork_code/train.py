import tensorflow as tf
import numpy as np
import sklearn
import random,os,io,pandas as pd, yaml,sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.callbacks import Callback, EarlyStopping, LearningRateScheduler, ModelCheckpoint
from hyperparameter import SobolevNetwork

if __name__=="__main__":
    dep_train=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\XTr.dat",sep='\s+',names=[str(i) for i in range(0,16)])
    indep_train=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\yTr.dat",sep='\s+',names=['target'])
    dep_val=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\XV.dat",sep='\s+',names=[str(i) for i in range(0,16)])
    indep_val=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\yV.dat",sep='\s+',names=["target"])
    
    with open("params.yaml", 'r') as stream:
        params = yaml.safe_load(stream)
    print("params",params)

    model= SobolevNetwork(input_dim=16,num_hidden1=params["layer_1"],num_hidden2=params["layer_2"],num_hidden3=params["layer_3"],num_hidden4=params["layer_4"],num_hidden5=params["layer_5"],num_hidden6=params["layer_6"],dropout=params["drop_out"])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                    loss='mean_squared_error', metrics=["mean_squared_error","mean_absolute_error","mean_absolute_percentage_error","mean_squared_logarithmic_error"])
    history=model.fit(dep_train, indep_train, batch_size=32, epochs=3, validation_data=(dep_val, indep_val))
    model.save("model")
    pd.DataFrame(history.history).to_csv("metrices.csv",index=False)