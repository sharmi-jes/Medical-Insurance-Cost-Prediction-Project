import sys
import os
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join("artifacts/preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()

    def get_data_transformation(self):
        logging.info("get the data transformation based on the data")
        numerical_cols=['age', 'bmi', 'children', ]
        categorical_cols=['sex', 'smoker', 'region']


        num_pipeline=Pipeline(
            steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
            ]
        )

        cat_pipeline=Pipeline(
            steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehotencoder",OneHotEncoder()),
                ("scaler",StandardScaler())
            ]
        )

        preprocessor=ColumnTransformer(
            [
                ("num_pipeline",num_pipeline,numerical_cols),
                ("cat_pipeline",cat_pipeline,categorical_cols)
            ]
        )


        return preprocessor
    
    
    def intiate_data_transformatio(self, train_data, test_data):
        """
        Transforms train and test data using the preprocessor and saves the preprocessor object.

        Args:
            train_data (str | pd.DataFrame): Path to train data CSV or DataFrame.
            test_data (str | pd.DataFrame): Path to test data CSV or DataFrame.

        Returns:
            Tuple: Processed train array, test array, and preprocessor object file path.
        """
        try:
            # Load train data
            if isinstance(train_data, str):
                assert os.path.exists(train_data), f"Train file not found: {train_data}"
                logging.info(f"Reading train data from: {train_data}")
                train_df = pd.read_csv(train_data)
            elif isinstance(train_data, pd.DataFrame):
                train_df = train_data
            else:
                raise ValueError("train_data must be a file path or a DataFrame")

            # Load test data
            if isinstance(test_data, str):
                assert os.path.exists(test_data), f"Test file not found: {test_data}"
                logging.info(f"Reading test data from: {test_data}")
                test_df = pd.read_csv(test_data)
            elif isinstance(test_data, pd.DataFrame):
                test_df = test_data
            else:
                raise ValueError("test_data must be a file path or a DataFrame")

            logging.info("Creating preprocessor object")



            target_column_name="charges"
            logging.info("take the preprocessor for the transfer train and test data as a same format")
            preprocessor_obj=self.get_data_transformation()
             
            logging.info("drop the traget col from the train and test") 
            input_train_data_path=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_test_data_path=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("apply preprocessor obj for input triana nd test data")
            preprocessor_obj_train_path=preprocessor_obj.fit_transform(input_train_data_path)
            preprocessor_obj_test_path=preprocessor_obj.transform(input_test_data_path)
            
            logging.info("combine the targeta nd inpendent features")
            train_arr=np.c_[
                preprocessor_obj_train_path,np.array(target_feature_train_df),
                
            ]
            test_arr=np.c_[
                preprocessor_obj_test_path,np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
                
            )
        except Exception as e:
            raise CustomException(e,sys)
