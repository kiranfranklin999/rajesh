### Hyperparameter tuning

## How to run 

`python hyperparameter.py`

## What can be expected?

1. a yaml file will be generated with best parameters [params.yaml]

## what things you can changes?

1. Number epochs and batch size in line 108 [ epochs=20,batch_size=32]
2. Under `model_builder()` you can change the range of min and max neurons, dropout list and learning rate.


### Training with best params

## How to run?

`python train.py`

## what can be expected ?

1. model will be save with name "model"
2. metrices.csv file will be generated having val and training info which can be used to plot graphs aor further analysis

## what things you can changes?

1. Number of epochs and batch size [batch_size=32, epochs=3] in line 23


### predict new data

## How to run?

`python predict.py`

## what can be expected ?

1. predicted_data.csv file with test and predicted data. 
