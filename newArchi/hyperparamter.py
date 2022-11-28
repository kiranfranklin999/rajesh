import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

def build_model(hp):
    model = keras.Sequential()
    hp_activation_1 = hp.Choice('activation_2', values=['relu', 'tanh'])
    hp_activation_2 = hp.Choice('activation_3', values=['relu', 'tanh'])
    hp_activation_3 = hp.Choice('activation_4', values=['relu', 'tanh'])
    kwargs = {'hp_activation_0':hp_activation_1,'hp_activation_1':hp_activation_2,'hp_activation_2':hp_activation_3}
    hp_weights=hp.Choice('weights', values=['glorot_uniform','he_normal','he_uniform','random_uniform','random_normal'])
    for i,r in enumerate(range(hp.Int('num_layers', 2, 10))):
        print(i,r)
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=2,
                                            max_value=10,
                                            step=2),kernel_initializer=hp_weights,
                               activation=kwargs[f'hp_activation_{i}']))
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
    return best_params,best_model

if __name__=="__main__":
    dep_train=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\XTr.dat",sep='\s+',names=[str(i) for i in range(0,16)])
    indep_train=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\yTr.dat",sep='\s+',names=['target'])
    dep_val=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\XV.dat",sep='\s+',names=[str(i) for i in range(0,16)])
    indep_val=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\yV.dat",sep='\s+',names=["target"])
    dep_test=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\XT.dat",sep='\s+',names=[str(i) for i in range(0,16)])
    indep_test=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\yT.dat",sep='\s+',names=["target"])

    best_params,best_model=tune_model(train_x=dep_train,train_y=indep_train,val_x=dep_val,val_y=indep_val,epochs=20,batch_size=32,project_name="soblenetwork_tuning")
    with io.open('params.yaml', 'w', encoding='utf8') as outfile:
        yaml.dump(best_params[0].values, outfile, default_flow_style=False, allow_unicode=True)
    print("file_name_ids_inside form:",best_params[0].values,file=sys.stderr)