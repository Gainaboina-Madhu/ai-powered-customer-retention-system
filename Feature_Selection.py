import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import sys
import seaborn as sns
from numpy.distutils.conv_template import parse_values

from scipy.stats import yeojohnson
from seaborn import boxplot
from scipy import stats
from scipy.stats import boxcox


#modularization
import logging
from logging_code import setup_logging
logger = setup_logging('Feature_Selection')

import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

def Filter_Methods(X_train_num_cols, X_test_num_cols, y_train, y_test):
    try:
        logger.info(f'=================== Constant Technique =============================')
        logger.info(f'Before apply Train numerical columns and shapes variable Transformation \n : {X_train_num_cols.columns} : {X_train_num_cols.shape}')

        logger.info(f'Before apply Test numerical columns and shapes variable Transformation \n : {X_test_num_cols.columns} : {X_test_num_cols.shape}')

        # Applying the constant technique with threshold = 0
        var = VarianceThreshold(threshold=0)
        var.fit(X_train_num_cols)
        logger.info(f'Checking the Train Good Column {sum(var.get_support())} : {X_train_num_cols.columns[var.get_support()]}')
        logger.info(f'Checking the Train Bad Column {sum(~var.get_support())} : {X_train_num_cols.columns[~var.get_support()]}')
        logger.info(f'Checking the Test Good Column {sum(var.get_support())} : {X_test_num_cols.columns[var.get_support()]}')
        logger.info(f'Checking the Test Bad Column {sum(~var.get_support())} : {X_test_num_cols.columns[~var.get_support()]}')


        logger.info(f'=================== Quasi Constant Technique ================================')

        # Applying the Quasi Constant technique with threshold = 1(0.01)
        var = VarianceThreshold(threshold=0.01)
        var.fit(X_train_num_cols)
        logger.info(f'Checking the Train Good Column {sum(var.get_support())} : {X_train_num_cols.columns[var.get_support()]}')
        logger.info(f'Checking the Train Bad Column {sum(~var.get_support())} : {X_train_num_cols.columns[~var.get_support()]}')
        logger.info(f'Checking the Test Good Column {sum(var.get_support())} : {X_test_num_cols.columns[var.get_support()]}')
        logger.info(f'Checking the Test Bad Column {sum(~var.get_support())} : {X_test_num_cols.columns[~var.get_support()]}')

        logger.info(f'=================== Hypothesis Testing Technique =============================')

        # Applying the Hypothesis Testing Technique
        # Applying the pearson Technique
        logger.info(f'the X_train values {X_train_num_cols.shape}')
        logger.info(f'the y_train values {y_train.shape}')

        c = []
        for i in X_train_num_cols.columns:
            result = pearsonr(X_train_num_cols[i], y_train)
            c.append(result)

        t = np.array(c)

        p_values = pd.Series(t[:,1],index=X_train_num_cols.columns)

        p = 0
        f = []

        for i in p_values:
            if i < 0.05:
                f.append(X_train_num_cols.columns[p])
            p = p + 1

        logger.info(f'Checking the Good Columns: {f}')

        return X_train_num_cols,X_test_num_cols


    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

