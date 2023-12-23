import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_con_fig=ModelTrainerConfig()

    def initiate_model_tranier(self,train_array,test_array):
        try:
            logging.info("spliting training and test data")
            X_train,y_train,X_test,y_test = (
                                                train_array[:,:-1],
                                                train_array[:,-1],
                                                test_array[:,:-1],
                                                test_array[:,-1]
                                              
                                              
                                              )
            

            # create a dictanary of model
            models={
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()

            }

            

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,
                                             y_test=y_test,models=models)


            ## to get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            ## to get the best model name form the dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score <0.6 :
                raise CustomException("No best model found")
            
            logging.info("best model found on both the train and test dataset")

            save_object(
                file_path=self.model_trainer_con_fig.train_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test,predicted)

            return r2_square



        except Exception as e:
            raise CustomException(e,sys)