from logging import info

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import sys

#importings
from sklearn.model_selection import train_test_split


#modularization
import logging
from logging_code import setup_logging
logger = setup_logging('main')   #creating a new logging file for main

from handling_missing_values_mode import handling_missing  #calling the missing values file
from Variable_Transformation import Var_Transforamtion
from Feature_Selection import Filter_Methods
from Outliers import outlier
from Categorical_to_Numerical import Cat_to_Num
from Data_Balance import Data_Balancer
from Feature_Scaling import feature_scale
from All_Models_code import common
from Hyperparameter_Tuning import tuning


class CHURN:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path) #loading the dataset into df
            logger.info(self.df)
            logger.info(f'{self.df.info()}')


            logger.info(f'Before Updated dataset Size is: {self.df.shape}')

            # Adding a new column "sim" based on the Internet Service

            def add_sim(df):
                if df['PaymentMethod'] == 'Electronic check':
                    return 'Reliance Jio'
                elif df['PaymentMethod'] == 'Mailed check':
                    return 'Airtel'
                elif df['PaymentMethod'] == 'Bank transfer (automatic)':
                    return 'Vi-idea'
                else:
                    return 'BSNL'
            self.df['Sim'] = self.df.apply(add_sim,axis=1)

            logger.info(f'After updated file is {self.df}')
            logger.info(f'After updated dataset Size is: {self.df.shape}')
            logger.info(f'After updated dataset Size is: {self.df.columns}')

            logger.info(f'Checking for null values')
            for i in self.df.columns:
                logger.info(f'{i} -> {self.df[i].isnull().sum()}')


            logger.info(f'===================== converting the object type into float ============================')

            # converting the total charge's into numeric
            self.df['TotalCharges'] = self.df['TotalCharges'].replace(" ", np.nan)
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'])
            logger.info(f'{self.df.info()}')

            for i in self.df.columns:
                logger.info(f'{i} -> {self.df[i].isnull().sum()}')


            #Divide the data into independent(X) and dependent(y)
            self.X = self.df.drop('Churn',axis=1)
            self.y = self.df['Churn']

            logger.info(f'checking the column names X : {self.X.columns}')
            logger.info(f'checking the shape of y : {self.y.shape}')

            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=45)

            #here converting the actual point from string(Yes/No) to numeric(1/0)
            self.y_train = self.y_train.map({'Yes':0,'No':1}).astype(int)
            self.y_test = self.y_test.map({'Yes': 0, 'No': 1}).astype(int)
            logger.info(f'Train data size {self.X_train.shape} and {self.y_train.shape}')
            logger.info(f'Test data size {self.X_test.shape} and {self.y_test.shape}')
            logger.info(self.y_train)
            logger.info(self.y_test)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def missing_values(self):
        try:
            logger.info(f'==================  Handling missing Values  =============================')
            logger.info(f'Before replacing Train null values : {self.X_train.isnull().sum()}')
            logger.info(f'Before replacing Test null values : {self.X_test.isnull().sum()}')

            self.X_train,self.X_test=handling_missing(self.X_train,self.X_test)

            logger.info(f'After replacing Train null values : {self.X_train.isnull().sum()}')
            logger.info(f'After replacing Test null values : {self.X_test.isnull().sum()}')


        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def data_seperation(self):
        try:
            logger.info(f'================ Data Splitting =======================================')
            logger.info(f'Before splitting the Train columns {self.X_train.columns}')
            logger.info(f'Before splitting the Test columns {self.X_test.columns}')

            self.X_train_nums_cols = self.X_train.select_dtypes(exclude=object)
            self.X_train_cats_cols = self.X_train.select_dtypes(include=object)

            self.X_test_nums_cols = self.X_test.select_dtypes(exclude=object)
            self.X_test_cats_cols = self.X_test.select_dtypes(include=object)

            logger.info(f'After splitting the Train numerical columns {self.X_train_nums_cols.columns} : \n {self.X_train_nums_cols.shape}')
            logger.info(f'After splitting the Train categorical  columns {self.X_train_cats_cols.columns} : \n {self.X_train_cats_cols.shape}')
            logger.info(f'After splitting the Test numerical columns {self.X_test_nums_cols.columns} : \n {self.X_test_nums_cols.shape}')
            logger.info(f'After splitting the Test categorical  columns {self.X_test_cats_cols.columns} : \n {self.X_test_cats_cols.shape}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def Variable_Tran(self):
        try:
            logger.info(f'Before apply Train numerical columns and shapes variable Transformation \n : {self.X_train_nums_cols.columns} : {self.X_train_nums_cols.shape}')
            logger.info(f'Before apply Test numerical columns and shapes variable Transformation \n : {self.X_test_nums_cols.columns} : {self.X_test_nums_cols.shape}')

            self.X_train_nums_cols,self.X_test_nums_cols = Var_Transforamtion(self.X_train_nums_cols,self.X_test_nums_cols)

            logger.info(f'After apply Train numerical columns and shapes variable Transformation \n : {self.X_train_nums_cols.columns} : {self.X_train_nums_cols.shape}')
            logger.info(f'After apply Test numerical columns and shapes variable Transformation \n : {self.X_test_nums_cols.columns} : {self.X_test_nums_cols.shape}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def out_liers(self):
        try:
            logger.info(f'Before apply Train numerical columns and shapes outlier \n : {self.X_train_nums_cols.columns} : {self.X_train_nums_cols.shape}')
            logger.info(f'Before apply Test numerical columns and shapes oulier \n : {self.X_test_nums_cols.columns} : {self.X_test_nums_cols.shape}')

            self.X_train_nums_cols, self.X_test_nums_cols = outlier(self.X_train_nums_cols, self.X_test_nums_cols)

            logger.info(f'After apply Train numerical columns and shapes outlier \n : {self.X_train_nums_cols.columns} : {self.X_train_nums_cols.shape}')
            logger.info(f'After apply Test numerical columns and shapes outlier \n : {self.X_test_nums_cols.columns} : {self.X_test_nums_cols.shape}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def feature_selections(self):
        try:
            self.X_train_nums_cols,self.X_test_nums_cols = Filter_Methods(self.X_train_nums_cols,self.X_test_nums_cols,self.y_train,self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def Categorical_Numerical(self):
        try:

            self.X_train_cats_cols,self.X_test_cats_cols = Cat_to_Num(self.X_train_cats_cols,self.X_test_cats_cols)

            self.X_train_nums_cols.reset_index(drop=True,inplace=True)
            self.X_train_cats_cols.reset_index(drop=True,inplace=True)
            self.X_test_nums_cols.reset_index(drop=True,inplace=True)
            self.X_test_cats_cols.reset_index(drop=True,inplace=True)

            self.Training_data = pd.concat([self.X_train_nums_cols,self.X_train_cats_cols],axis=1)
            self.Testing_data = pd.concat([self.X_test_nums_cols,self.X_test_cats_cols],axis=1)

            logger.info(f'Total Training Data is : {self.Training_data.columns} \n and it shape : {self.Training_data.shape} ')
            logger.info(f'Total Testing Data is : {self.Testing_data.columns} \n and it shape : {self.Testing_data.shape}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def Data_Balancing(self):
        try:
            logger.info(f'========================= Data Balancing ================================')

            logger.info(f'Before Data Balanced the {self.Training_data.columns} and it is shape {self.Training_data.shape}')
            self.Training_data_bal, self.y_train_bal = Data_Balancer(self.Training_data,self.Testing_data,self.y_train,self.y_test)
            logger.info(f'After balancing the data {self.Training_data_bal.columns} and it is shape : {self.Training_data_bal.shape}')
            logger.info(f' After balancing the data {self.y_train_bal.shape}')


        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def feature_scaler(self):
        try:
            logger.info(f' Before applying the feature scaling Training data {self.Training_data_bal.columns} and its shape {self.Training_data_bal.shape}')
            logger.info(f' Before applying the feature scaling Testing data {self.Testing_data} and its shape {self.Testing_data.shape} ')

            feature_scale(self.Training_data_bal, self.y_train_bal, self.Testing_data,self.y_test)

            #logger.info(f' After applying the feature scaling Training data {self.Training_data_bal_sc} {self.Training_data_bal_sc.shape}')
            #logger.info(f' After applying the feature scaling Testing data {self.Testing_data_sc.shape}')


        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')




    def all_models(self):
        try:
            self.Training_data_bal_sc,self.y_train_bal,self.Testing_data_sc,self.y_test = common(self.Training_data_bal_sc,self.y_train_bal,self.Testing_data_sc,self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def hyperparameter(self):
        try:
            tuning(self.Training_data_bal_sc,self.y_train_bal,self.Testing_data_sc,self.y_test )

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')


if __name__ == '__main__':
    obj = CHURN('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    obj.missing_values()
    obj.data_seperation()
    obj.Variable_Tran()
    obj.out_liers()
    obj.feature_selections()
    obj.Categorical_Numerical()
    obj.Data_Balancing()
    obj.feature_scaler()
    #obj.all_models()
    #obj.hyperparameter()

