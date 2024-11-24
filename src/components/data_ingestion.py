import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join("artifacts/train.csv")
    test_data_path:str=os.path.join("artifacts/test.csv")
    raw_data_path:str=os.path.join("artifacts/raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def  initiate_data_ingestion(self):
        try:
            logging.info("read the data")
            df=pd.read_csv(r"D:\RESUME ML PROJECTS\Medical Insuarnce\notebooks\cleaned.csv")
            logging.info("create a diredtocry")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            logging.info("pass the raw data to that particular path")
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)


            logging.info("split the data as train and test data")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path)

            test_set.to_csv(self.ingestion_config.test_data_path)

            return(
                train_set,
                test_set
            )
        
        except Exception as e:
            raise CustomException(e,sys)
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    transformation = DataTransformation()
    train_data_path = obj.ingestion_config.train_data_path
    test_data_path = obj.ingestion_config.test_data_path
    transformation.intiate_data_transformatio(train_data_path, test_data_path)
