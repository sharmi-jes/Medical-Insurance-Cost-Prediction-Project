import sys
import os
from src.exception import CustomException
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pickle


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
        
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

# def evaluate_model(x_train, y_train, x_test, y_test, models, params):
#     try:
#         report = {}
#         for model_name, model in models.items():
#             param_grid = params.get(model_name, {})  # Get params for the specific model

#             # Initialize GridSearchCV with model and its parameters
#             gs = GridSearchCV(model, param_grid, cv=3)
#             gs.fit(x_train, y_train)

#             # Set best parameters and train model
#             model.set_params(**gs.best_params_)
#             model.fit(x_train, y_train)

#             # Predictions and scoring
#             y_train_pred = model.predict(x_train)
#             y_test_pred = model.predict(x_test)
#             r2_score_train = r2_score(y_train, y_train_pred)
#             r2_score_test = r2_score(y_test, y_test_pred)

#             # Save the test score in the report
#             report[model_name] = r2_score_test

#         return report  # Return after evaluating all models

#     except Exception as e:
#         raise CustomException(e, sys)

