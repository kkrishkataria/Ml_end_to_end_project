import sys,os
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass
from src.utils import save_obj

@dataclass
class DataTransformConfig:
    preprocessor_obj_file:str=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_trans_config=DataTransformConfig()
    def get_data_trans_obj(self):  ## getting column transformer object 
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
                ]
            num_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='median')),
                    ("Scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder",OneHotEncoder())
                ]
            )
            logging.info('Scaling of Numerical Features done Successfully !')
            logging.info('Encoding of Categorical Features done Successfully !')

            preprocessor=ColumnTransformer(
                transformers=[
                ('Numerical_Pipeline',num_pipeline,numerical_columns),
                ('Categorical_Pipeline',cat_pipeline,categorical_columns)]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Train and Test data readed Successfully !')
            logging.info('Obtaining Preproceesor object !')

            preprocessing_obj=self.get_data_trans_obj()
            target_column="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_features_train_df=train_df.drop(columns=target_column,axis=1)
            target_feature_train_df=train_df[target_column]
            input_features_test_df=test_df.drop(columns=target_column,axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info('Applying Preproceesor object on Training and Test Dataframes !')

            input_features_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessing_obj.transform(input_features_test_df)

            train_arr=np.c_[ # for concatination 
                input_features_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[ # for concatination 
                input_features_test_arr,np.array(target_feature_test_df)
            ]

            logging.info('Saved Preprocesed Object !')

            save_obj(
                filepath=self.data_trans_config.preprocessor_obj_file,
                obj=preprocessing_obj
            )
            return (
                train_arr,test_arr
            )
        except Exception as e:
            raise CustomException(e,sys)
