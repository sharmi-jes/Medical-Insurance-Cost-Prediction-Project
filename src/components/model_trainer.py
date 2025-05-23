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
import mlflow
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    model_file_path:str=os.path.join("artifacts/model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.trainer_config=ModelTrainerConfig()


#     # def track_mlflow(model):
#     #     with mlflow.start_run():
#     #         mlflow.log_metrics()

#     def initiate_model_trainer(self,train_array,test_array):
#         try:
#             logging.info("split the train and test data as x_train,y_train,x_test,y_tets")
#             x_train,y_train,x_test,y_test=(
#                 train_array[:,:-1],
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1]
#             )

#             logging.info("take the models")
#             models={
#                 "LinearRegression":LinearRegression(),
#                 # "SVR":SVR(),
#                 "RandomForestRegressor":RandomForestRegressor(),
#                 "AdaBoostRegressor":AdaBoostRegressor(),
#                 "GradientBoostingRegressor":GradientBoostingRegressor(),
#                 "DecisionTreeRegressor":DecisionTreeRegressor()
#             }

#             params = {
#                   "LinearRegression": {
#         "fit_intercept": True,
#         "normalize": False,          # deprecated in latest sklearn, use StandardScaler instead
#         "copy_X": True,
#         "n_jobs": None
#     },
#                "RandomForestRegressor": {
#         "n_estimators": 100,
#         "criterion": "squared_error",  # formerly "mse"
#         "max_depth": None,
#         "min_samples_split": 2,
#         "min_samples_leaf": 1,
#         "max_features": "auto",
#         "bootstrap": True,
#         "n_jobs": -1,
#         "random_state": 42,
#         "verbose": 0
#     },
#                 "AdaBoostRegressor": {
#         "base_estimator": None,       # default DecisionTreeRegressor(max_depth=3)
#         "n_estimators": 50,
#         "learning_rate": 1.0,
#         "loss": "linear",
#         "random_state": 42
#     },
#                "GradientBoostingRegressor": {
#         "loss": "squared_error",
#         "learning_rate": 0.1,
#         "n_estimators": 100,
#         "subsample": 1.0,
#         "criterion": "friedman_mse",
#         "min_samples_split": 2,
#         "min_samples_leaf": 1,
#         "max_depth": 3,
#         "random_state": 42,
#         "verbose": 0
#     },
#                 "DecisionTreeRegressor": {
#         "criterion": "squared_error",
#         "splitter": "best",
#         "max_depth": None,
#         "min_samples_split": 2,
#         "min_samples_leaf": 1,
#         "random_state": 42
#     }
# }



             
#             model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models,params)
#             logging.info("calculate best score")
#             best_score=max(sorted(model_report.values()))

#             best_name=list(model_report.keys())[
#                 list(model_report.values()).index(best_score)
#             ]
#             logging.info("take the best model")
#             best_model=models[best_name]

#             save_object(
#                 file_path=self.trainer_config.model_file_path,
#                 obj=best_model
#             )
#             logging.info("check the relation ")
#             if best_score<0.6:
#                 raise CustomException("Not found a good model")
#             logging.info("we get best score both on train and test data")
#             logging.info("predicted the x_test data")
#             prediction=best_model.predict(x_test)
#             logging.info(f"r2_score value")
#             score_performance=r2_score(y_test,prediction)
#             print(score_performance)
#             return score_performance
#         except Exception as e:
#             raise CustomException(e,sys)
import sys
import pickle
import mlflow
import mlflow.sklearn
import logging

from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.utils import save_object
# from src.config.configuration import ModelTrainerConfig

@dataclass
class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split the train and test data as x_train, y_train, x_test, y_test")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Initialize models")
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor()
            }

            param = {
                "LinearRegression": [
                    {
                        "fit_intercept": [True, False],
                        "copy_X": [True]
                    }
                ],
                "RandomForestRegressor": [
                    {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10],
                        "random_state": [42]
                    }
                ],
                "AdaBoostRegressor": [
                    {
                        "n_estimators": [50, 100],
                        "learning_rate": [1.0, 0.1],
                        "random_state": [42]
                    }
                ],
                "GradientBoostingRegressor": [
                    {
                        "n_estimators": [100, 150],
                        "learning_rate": [0.1],
                        "max_depth": [3, 5],
                        "random_state": [42]
                    }
                ],
                "DecisionTreeRegressor": [
                    {
                        "max_depth": [None, 5, 10],
                        "min_samples_split": [2, 5],
                        "random_state": [42]
                    }
                ]
            }

            model_report, best_model_objects = evaluate_model(x_train, y_train, x_test, y_test, models, param)
            logging.info("Model evaluation completed.")

            best_score = max(model_report.values())
            best_name = list(model_report.keys())[list(model_report.values()).index(best_score)]
            best_model, best_params = best_model_objects[best_name]

            logging.info(f"Best model: {best_name} with score {best_score}")

            # ---------- MLflow logging ----------
            with mlflow.start_run(run_name=f"Model_Training_{best_name}"):
                mlflow.log_params(best_params)
                mlflow.log_metric("r2_score", best_score)
                # mlflow.sklearn.log_model(best_model, artifact_path="model")
                logging.info(f"Logged {best_name} model to MLflow")
            # ---------- End MLflow logging ----------

            save_object(
                file_path=self.trainer_config.model_file_path,
                obj=best_model
            )

            prediction = best_model.predict(x_test)
            score_performance = r2_score(y_test, prediction)
            logging.info(f"Final R2 score: {score_performance}")

            if best_score < 0.6:
                raise CustomException("Not found a good model", sys)

            return score_performance

        except Exception as e:
            raise CustomException(e, sys)
