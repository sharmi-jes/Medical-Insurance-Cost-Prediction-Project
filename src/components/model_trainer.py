import sys
import os
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from dataclasses import dataclass

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    model_file_path:str=os.path.join("artifacts/model.pkl")

class ModelTrainer:
    def __init__(self):
        self.trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split the train and test data as x_train,y_train,x_test,y_tets")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info("take the models")
            models={
                "LinearRegression":LinearRegression(),
                "SVR":SVR(),
                "RandomForestRegressor":RandomForestRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor()
            }
             
            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)
            logging.info("calculate best score")
            best_score=max(sorted(model_report.values()))

            best_name=list(model_report.keys())[
                list(model_report.values()).index(best_score)
            ]
            logging.info("take the best model")
            best_model=models[best_name]

            save_object(
                file_path=self.trainer_config.model_file_path,
                obj=best_model
            )
            logging.info("check the relation ")
            if best_score<0.6:
                raise CustomException("Not found a good model")
            logging.info("we get best score both on train and test data")
            logging.info("predicted the x_test data")
            prediction=best_model.predict(x_test)
            logging.info(f"r2_score value")
            score_performance=r2_score(y_test,prediction)
            print(score_performance)
            return score_performance
        except Exception as e:
            raise CustomException(e,sys)