import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import sys
import seaborn as sns
import logging

import pickle

import warnings
warnings.filterwarnings('ignore')


#modularization

from logging_code import setup_logging
logger = setup_logging('Feature_Scaling')


from sklearn.preprocessing import StandardScaler #Z_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


def feature_scale(Training_data_bal,y_train_bal, Testing_data, y_test):
    try:
        logger.info(f' Before applying the feature scaling Training data {Training_data_bal.columns} and its shape {Training_data_bal.shape}')
        logger.info(f' Before applying the feature scaling Testing data {Testing_data}')

        sc = StandardScaler()
        sc.fit(Training_data_bal)

        Training_data_bal_sc = sc.transform(Training_data_bal)
        Testing_data_sc = sc.transform(Testing_data)

        with open('standard_scaler.pkl', 'wb') as f:
            pickle.dump(sc, f)


        lr_reg = LogisticRegression(C=1,class_weight=None,max_iter=100,penalty='l2',solver='sag')
        lr_reg.fit(Training_data_bal, y_train_bal)
        predictions = lr_reg.predict(Testing_data)
        logger.info(f'confusion matrix : \n {confusion_matrix(y_test, predictions)}')
        logger.info(f'Accuracy score : {accuracy_score(y_test, predictions)}')
        logger.info(f'Classification report : \n {classification_report(y_test, predictions)}')


        with open('Model.pkl', 'wb') as t:
            pickle.dump(lr_reg, t)




    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')