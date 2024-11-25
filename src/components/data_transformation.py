import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.utils import save_object
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join("artifacts/preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()
    logging.info("get data transformation")
    def get_data_transformation(self):
        numerical_cols=['age', 'bmi', 'children']
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
                ("onehotencoder",OneHotEncoder())
            ]
        )

        preprocessor=ColumnTransformer(
            [
                ("num_pipeline",num_pipeline,numerical_cols),
                ("cat_pipeline",cat_pipeline,categorical_cols)
            ]
        )
        logging.info("return the preprocessor") 
        return preprocessor
    
    
    def initiate_data_transformation(self, train_data, test_data):
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


            target_col="charges"

            preprocessor_obj=self.get_data_transformation()
            logging.info("drop target col from the train andn test data") 
            input_train_df=train_df.drop(columns=[target_col])
            input_target=train_df[target_col]

            input_test_df=test_df.drop(columns=[target_col])
            input_target_test=test_df[target_col]
            logging.info("apply preprocessor for the train and test data")
            input_preprocessor_train=preprocessor_obj.fit_transform(input_train_df)
            input_preprocessor_test=preprocessor_obj.transform(input_test_df)
            logging.info("combine the input data and target col in the train and test data")
            train_array=np.c_[
                input_preprocessor_train,np.array(input_target)
            ]

            test_array=np.c_[
                input_preprocessor_test,np.array(input_target_test)
            ]

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_array,
                test_array,
                self.transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)

       