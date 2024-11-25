import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
from sklearn.preprocessing import StandardScaler,OneHotEncoder

class PredictPipeline:
    def __init__(self):
        pass
    logging.info("create a predict function to predict the data")
    def predict(self,features):
        try:
            model_file="artifacts/model.pkl"
            preprocessor_file="artifacts/preprocessor.pkl"
            model=load_object(file_path=model_file)
            preprocessor=load_object(file_path=preprocessor_file)
            data_sclaed=preprocessor.transform(features)
            prediction=model.predict(data_sclaed)
            print(prediction)
            return prediction
        except Exception as e:
            raise CustomException(e,sys)

#   numerical_cols=['age', 'bmi', 'children']
        # categorical_cols=['sex', 'smoker', 'region']
class CustomData:
    def __init__(self,age,bmi,children,sex,smoker,region):
        self.age=age
        self.bmi=bmi
        self.children=children
        self.sex=sex
        self.smoker=smoker
        self.region=region
    logging.info("create a get data as daatframe bcoz we can pass the datafarme to the model")
    def get_data_as_datagrame(self):
        try:
            input_data={
                'age':[self.age],
                'bmi':[self.bmi],
                'children':[self.children],
                'sex':[self.sex],
                'smoker':[self.smoker],
                'region':[self.region],

            }

            return pd.DataFrame(input_data)
        except Exception as e:
            raise CustomException(e,sys)

