import tensorflow as tf
import pandas as pd
import numpy as np

if __name__=="__main__":
    dep_test=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\XT.dat",sep='\s+',names=[str(i) for i in range(0,16)])
    indep_test=pd.read_csv("D:\\bro\\Analysis\\mSANN\\cd\\yT.dat",sep='\s+',names=["target"])
    model=tf.keras.models.load_model("model")
    predicted_data=model.predict(dep_test)
    predict=np.concatenate(predicted_data)
    pd.DataFrame(zip(indep_test['target'],predict),columns=['test_y','predict_y']).to_csv("predicted_data.csv",index=False)