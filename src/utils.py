import os,sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import pickle

def save_obj(filepath,obj):
    try:
        dirname=os.path.dirname(filepath)
        os.makedirs(dirname,exist_ok=True)
        with open(filepath,'wb') as file:
            pickle.dump(obj,file)

    except Exception as e:
        raise CustomException(e,sys)
