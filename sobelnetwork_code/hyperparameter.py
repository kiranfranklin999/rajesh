import tensorflow as tf
import numpy as np
import sklearn
import random,os,io,pandas as pd, yaml,sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.callbacks import Callback, EarlyStopping, LearningRateScheduler, ModelCheckpoint,CSVLogger
from tensorboard.plugins.hparams import api as hp
from keras_tuner import RandomSearch

class SobolevNetwork(Model):
    def __init__(self, input_dim: int, num_hidden1: int,num_hidden2: int,num_hidden3: int,num_hidden4: int,num_hidden5: int,num_hidden6: int,dropout: int,init = None):
        super(SobolevNetwork, self).__init__()
        self.input_dim = input_dim 
       

        self.W1 = tf.Variable(tf.random.normal([self.input_dim, num_hidden1],stddev=0.1))
        self.b1 = tf.Variable(tf.ones([num_hidden1]))
        self.dp1 = tf.keras.layers.Dropout(dropout)
        self.W2 = tf.Variable(tf.random.normal([num_hidden1, num_hidden2],stddev=0.1))
        self.b2 = tf.Variable(tf.ones([num_hidden2]))
        self.dp2 = tf.keras.layers.Dropout(dropout)
        self.W3 = tf.Variable(tf.random.normal([num_hidden2, num_hidden3],stddev=0.1))
        self.b3 = tf.Variable(tf.ones([num_hidden3]))
        self.dp3 = tf.keras.layers.Dropout(dropout)
        self.W4 = tf.Variable(tf.random.normal([num_hidden3, num_hidden4],stddev=0.1))
        self.b4 = tf.Variable(tf.ones([num_hidden4]))
        self.dp4 = tf.keras.layers.Dropout(dropout)
        self.W5 = tf.Variable(tf.random.normal([num_hidden4, num_hidden5],stddev=0.1))
        self.b5 = tf.Variable(tf.ones([num_hidden5]))
        self.dp5 = tf.keras.layers.Dropout(dropout)
        self.W6 = tf.Variable(tf.random.normal([num_hidden5, num_hidden6],stddev=0.1))
        self.b6 = tf.Variable(tf.ones([num_hidden6]))
        self.dp6 = tf.keras.layers.Dropout(dropout)        
        self.W7 = tf.Variable(tf.random.normal([num_hidden6, 1],stddev=0.1))
        self.b7 = tf.Variable(tf.ones([1]))
        self.w = [(self.W1, self.b1), (self.W2, self.b2), (self.W3, self.b3),(self.W4, self.b4), (self.W5, self.b5), (self.W6, self.b6),(self.W7, self.b7)]
        
    def call(self, X):
        #Input layer
        out = X
        #Hidden layers
        W,b = self.w[0]
        out = tf.nn.tanh(tf.matmul(out, W) + b)
        out = self.dp1(out)
        W,b = self.w[1]
        out = tf.nn.tanh(tf.matmul(out, W) + b)
        out = self.dp2(out)
        W,b = self.w[2]
        out = tf.nn.sigmoid(tf.matmul(out, W) + b)
        out = self.dp3(out)
        W,b = self.w[3]
        out = tf.nn.sigmoid(tf.matmul(out, W) + b)
        out = self.dp4(out)
        W,b = self.w[4]
        out = tf.nn.leaky_relu(tf.matmul(out, W) + b)
        out = self.dp5(out)
        W,b = self.w[5]
        out = tf.nn.relu(tf.matmul(out, W) + b)
        #Output layer
        W,b = self.w[-1]
        out = tf.matmul(out, W) + b
        return out

def model_builder(hp):

  hp_layer_1 = hp.Int('layer_1', min_value=1, max_value=200, step=5)
  hp_layer_2 = hp.Int('layer_2', min_value=1, max_value=200, step=5)
  hp_layer_3 = hp.Int('layer_3', min_value=1, max_value=200, step=5)
  hp_layer_4 = hp.Int('layer_4', min_value=1, max_value=200, step=5)
  hp_layer_5 = hp.Int('layer_5', min_value=1, max_value=200, step=5)
  hp_layer_6 = hp.Int('layer_6', min_value=1, max_value=200, step=5)
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
  hp_drop_out = hp.Choice('drop_out', values=[0.1, 0.2, 0.15,0.5, 0.25,0.75])

  model= SobolevNetwork(input_dim=16,num_hidden1=hp_layer_1,num_hidden2=hp_layer_2,num_hidden3=hp_layer_3,num_hidden4=hp_layer_4,num_hidden5=hp_layer_5,num_hidden6=hp_layer_6,dropout=hp_drop_out)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='mean_squared_error', metrics=["mean_squared_error","mean_absolute_error","mean_absolute_percentage_error","mean_squared_logarithmic_error"])
  
  return model


def tune_model(train_x,train_y,val_x,val_y,epochs,batch_size,project_name):
    print(train_x.shape,train_y.shape,val_x.shape,val_y.shape,epochs,batch_size,project_name)
    es_callback = EarlyStopping( monitor='val_loss',min_delta=0,patience=10, verbose=1, mode="auto", baseline=None, restore_best_weights=False)
    tuner =  RandomSearch(hypermodel=model_builder,
                      objective="val_mean_squared_error",
                      #objective=Objective(name="val_mean_squared_error",direction="min"),
                      max_trials=5,
                      #seed=123,
                      project_name=project_name,
                      overwrite=True
                    )
    tuner.search(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y))
    best_params = tuner.get_best_hyperparameters()
    best_model = tuner.get_best_models()[0]
    return best_params,best_model


if __name__=="__main__":
    dep_train=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\XTr.dat",sep='\s+',names=[str(i) for i in range(0,16)])
    indep_train=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\yTr.dat",sep='\s+',names=['target'])
    dep_val=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\XV.dat",sep='\s+',names=[str(i) for i in range(0,16)])
    indep_val=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\yV.dat",sep='\s+',names=["target"])
    dep_test=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\XT.dat",sep='\s+',names=[str(i) for i in range(0,16)])
    indep_test=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\yT.dat",sep='\s+',names=["target"])
    print(dep_train.shape,indep_train.shape,dep_val.shape,indep_val.shape,dep_test.shape,indep_test.shape)
    best_params,best_model=tune_model(train_x=dep_train,train_y=indep_train,val_x=dep_val,val_y=indep_val,epochs=20,batch_size=32,project_name="soblenetwork_tuning")
    with io.open('params.yaml', 'w', encoding='utf8') as outfile:
        yaml.dump(best_params[0].values, outfile, default_flow_style=False, allow_unicode=True)
    print("file_name_ids_inside form:",best_params[0].values,file=sys.stderr)
    
    