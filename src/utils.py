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


# --------------------- Utility Functions ---------------------

def evaluate_model(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}
        best_model_objects = {}

        for model_name, model in models.items():
            # logging.info(f"Running GridSearchCV for {model_name}")
            param_grid = param[model_name]
            gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
            gs.fit(x_train, y_train)

            best_model = gs.best_estimator_

            y_test_pred = best_model.predict(x_test)
            r2_score_test = r2_score(y_test, y_test_pred)

            report[model_name] = r2_score_test
            best_model_objects[model_name] = (best_model, gs.best_params_)

        return report, best_model_objects

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
