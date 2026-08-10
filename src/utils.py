import os,sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import pickle

def save_obj(filepath,obj):
    try:
        dirname=os.path.dirname(filepath)
        os.makedirs(dirname,exist_ok=True)
        with open(filepath,'wb') as file:
            pickle.dump(obj,file)

    except Exception as e:
        raise CustomException(e,sys)


def load_obj(file_path):
    try:
         with open(file_path,'rb') as file:
              return pickle.load(file)
    except Exception as e:
         raise CustomException(e,sys)

     
def evaluate_model(x_train  ,x_test, y_train ,y_test,models,params):
        try:  
            report ={}
            for i in range(len(list(models))):
                  model=list(models.values())[i]
                  para= params[list(models.keys())[i]]
                  gs=GridSearchCV(model,para,cv=3);
                  gs.fit(x_train,y_train)
                  model.set_params(**gs.best_params_)
                  model.fit(x_train,y_train)
                  y_train_pred=model.predict(x_train)
                  y_test_pred=model.predict(x_test)
                  train_model_score =r2_score(y_train,y_train_pred)
                  test_model_score=r2_score(y_test,y_test_pred)
                  report[list(models.keys())[i]]=test_model_score

            return report
      
        except Exception as e:
            raise CustomException(e,sys)