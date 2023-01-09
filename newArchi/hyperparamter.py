import pandas as pd, io, yaml, sys
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

def build_model(hp):
    model = keras.Sequential()
    hp_activation_1 = hp.Choice('activation_1', values=['relu', 'tanh'])
    hp_activation_2 = hp.Choice('activation_2', values=['relu', 'tanh'])
    hp_activation_3 = hp.Choice('activation_3', values=['relu', 'tanh'])
    hp_activation_4 = hp.Choice('activation_4', values=['relu', 'tanh'])
    hp_activation_5 = hp.Choice('activation_5', values=['relu', 'tanh'])
    hp_activation_6 = hp.Choice('activation_6', values=['relu', 'tanh'])
    hp_activation_7 = hp.Choice('activation_7', values=['relu', 'tanh'])
    hp_activation_8 = hp.Choice('activation_8', values=['relu', 'tanh'])
    kwargs = {'hp_activation_1':hp_activation_1,'hp_activation_2':hp_activation_2,'hp_activation_3':hp_activation_3,'hp_activation_4':hp_activation_4,
            'hp_activation_5':hp_activation_5,'hp_activation_6':hp_activation_6,'hp_activation_7':hp_activation_7,'hp_activation_8':hp_activation_8}
    hp_weights=hp.Choice('weights', values=['glorot_uniform','he_normal','he_uniform','random_uniform','random_normal'])
    model.add(layers.Input(16))
    
    for i,r in enumerate(range(hp.Int('num_layers', 2, 8))):
        print(i,r)
        model.add(layers.Dense(units=hp.Int('units_' + str(i+1),
                                            min_value=2,
                                            max_value=10,
                                            step=2),kernel_initializer=hp_weights,
                               activation=kwargs[f'hp_activation_{i+1}']))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error'])
    return model

def tune_model(train_x,train_y,val_x,val_y,epochs: int,batch_size: int,project_name: str):
    tuner = RandomSearch(build_model,objective='val_mean_absolute_error',max_trials=5,executions_per_trial=1,project_name=project_name)
    tuner.search(train_x, train_y,
             epochs=epochs,
             batch_size=batch_size,
             validation_data=(val_x, val_y))
    best_params = tuner.get_best_hyperparameters()
    best_model = tuner.get_best_models()[0]
    return best_params

if __name__=="__main__":
    dep_train=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\XTr.dat",sep='\s+',names=[str(i) for i in range(0,16)])
    dep_train=dep_train.iloc[:10,]
    indep_train=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\yTr.dat",sep='\s+',names=['target'])
    indep_train=indep_train.iloc[:10,]
    dep_val=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\XV.dat",sep='\s+',names=[str(i) for i in range(0,16)])
    dep_val=dep_val.iloc[:10,]
    indep_val=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\yV.dat",sep='\s+',names=["target"])
    indep_val=indep_val.iloc[:10,]
    dep_test=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\XT.dat",sep='\s+',names=[str(i) for i in range(0,16)])
    dep_test=dep_test.iloc[:10,]
    indep_test=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\yT.dat",sep='\s+',names=["target"])
    indep_test=indep_test.iloc[:10,]

    best_params=tune_model(train_x=dep_train,train_y=indep_train,val_x=dep_val,val_y=indep_val,epochs=5,batch_size=32,project_name="soblenetwork_tuning1")
    with io.open('params1.yaml', 'w', encoding='utf8') as outfile:
        yaml.dump(best_params[0].values, outfile, default_flow_style=False, allow_unicode=True)
    print("file_name_ids_inside form:",best_params[0].values,file=sys.stderr)