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
logger = setup_logging('handling_missing_values_mode')

def handling_missing(X_train,X_test):
    try:
        logger.info(f'Before replacing Train null values : {X_train.isnull().sum()}')
        logger.info(f'Before replacing Test null values : {X_test.isnull().sum()}')
        logger.info(f'Before replacing Train column shape {X_train.shape}')
        logger.info(f'Before replacing Train column shape {X_train.columns}')
        logger.info(f'Before replacing Test column shape {X_test.columns}')

        for i in X_train.columns:
            if X_train[i].isnull().sum() > 0:
                X_train[i+'_rep'] = X_train[i].copy()
                X_test[i+ '_rep'] = X_test[i].copy()
                X_train[i+'_rep'] = X_train[i+'_rep'].fillna(X_train[i+'_rep'].mode()[0])
                X_test[i + '_rep'] = X_test[i + '_rep'].fillna(X_test[i + '_rep'].mode()[0])

                '''
                a = X_train[i+'_rep']
                b = X_test[i+'_rep']
                plt.figure(figsize=(5,3))
                plt.title('Normal Distribution')
                X_train[i].plot(kind='kde',color='blue')
                X_test[i].plot(kind='kde', color='black')
                X_train[i+'_rep'].plot(kind='kde',color='r')
                X_test[i+'_rep'].plot(kind='kde', color='yellow')
                plt.legend()
                plt.show()
                
               '''

                X_train = X_train.drop([i], axis=1)
                X_test = X_test.drop([i], axis=1)

        logger.info(f'After replacing Train null values : {X_train.isnull().sum()}')
        logger.info(f'After replacing Test null values : {X_test.isnull().sum()}')
        logger.info(f'After replacing Train column shape {X_train.shape}')
        logger.info(f'After replacing Train column shape {X_train.columns}')
        logger.info(f'After replacing Test column shape {X_test.columns}')

        return X_train,X_test
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

