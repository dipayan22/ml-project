import sys
from dataclasses import dataclass
import os

from src.utils import save_object

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConFig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.DataTransformation_config=DataTransformationConFig()

    def get_data_tranformer_object(self):
        try:
            numeric_feature=['reading_score', 'writing_score']
            categorical_features = [
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
            ]

            num_pipeline= Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            categorical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))

                ]
            )
            logging.info("Numerical column scaling Completed")

            logging.info("Categorical column encoding Completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numeric_feature),
                    ("categorical_pipeline",categorical_pipeline,categorical_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test Comleted")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_tranformer_object()

            target_column_name="math_score"


            # numeric_feature=['reading_score', 'writing_score']
            # categorical_features = [
            #     'gender', 
            #     'race_ethnicity', 
            #     'parental_level_of_education', 
            #     'lunch', 
            #     'test_preparation_course'
            # ]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training and test dataframe"
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            # np.c_[] concatenates arrays along second axis.
            # Similarly, np.r_[] concatenates arrays along first axis.

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.DataTransformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.DataTransformation_config.preprocessor_obj_file_path,
            )







        except Exception as e:
            raise CustomException(e,sys)
            