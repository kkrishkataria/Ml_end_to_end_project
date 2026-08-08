import os,sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_obj,evaluate_model

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Spliting Training and Test input data')
            x_train,x_test,y_train,y_test=(
                train_arr[:,:-1],test_arr[:,:-1],train_arr[:,-1],test_arr[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
            }
            model_report:dict=evaluate_model(x_train,x_test,y_train,y_test,models)
            best_model_score=max(sorted(list(model_report.values())))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            logging.info('Best Model is found on the data')

            save_obj(
                self.model_trainer_config.trained_model_file_path,best_model
            )

            predicted=best_model.predict(x_test)
            r2_score_model=r2_score(y_test,predicted)
            return r2_score_model
        except Exception as e:
            raise CustomException(e,sys)


