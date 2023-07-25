import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import BayesianRidge, OrthogonalMatchingPursuit, Ridge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import KFold, cross_val_score

from tools import split_merge_datasets
from tools import fill_na_num
from tools import fill_na_cat
from tools import unskew_numeric

class Preprocessor():
    '''
    Preprocessor for Kaggle Housing Price Datasets

    In : 
        training data
        testing data

    Out : 
        prepped training data
        training targets
        prepped test data 
    '''
    def __init__(self,):

        return

    def run(self, train_df, test_df):

        y, merged_train_df, test_ids = split_merge_datasets(train_df, test_df)

        # recast  MSSubClass
        merged_train_df['MSSubClass'] = merged_train_df['MSSubClass'].astype('object')


        # split dataframe into numeric and categorical columns
        numeric_df = merged_train_df.select_dtypes(exclude = 'object')
        categorical_df = merged_train_df.select_dtypes('object')


        # filling missing values
        numeric_df = fill_na_num(numeric_df)
        categorical_df = fill_na_cat(categorical_df)


        # feature Transformation
        numeric_df = unskew_numeric(numeric_df)
        categorical_df = pd.get_dummies(categorical_df, dtype = 'float')

        # scale numeric features
        scaler = StandardScaler()
        scaler.fit(numeric_df)

        numeric_df = pd.DataFrame(scaler.transform(numeric_df), index=numeric_df.index, columns=numeric_df.columns)

        #  target Transformation
        y_log = unskew_numeric(y)

        # merge numeric and categorical dfs
        numeric_df.reset_index(drop = True, inplace=True)
        categorical_df.reset_index(drop = True, inplace=True)

        merged_train_df = pd.concat([numeric_df, categorical_df], axis = 1)


        # split train and test
        train_df = merged_train_df.iloc[ 0:len(y_log) , :].copy()
        test_df = merged_train_df.iloc[ len(y_log): , :].copy()

        return train_df, y_log, test_df, test_ids

class Trainer():

    def __init__(self, ):
        
        self.models = {
            'Bayesian Ridge' : BayesianRidge(),
            'CatBoost' : CatBoostRegressor(),
            'Ridge' : Ridge(),
            'Orthogonal Matching Pursuit' : OrthogonalMatchingPursuit(),
        }

        return

    def run(self, train_df, y_log):

        y_log = y_log.values.flatten()

        for name, model in self.models.items():
            model.fit(train_df, y_log)
            print(f'{name} trained.')

        kf = KFold(n_splits = 10)
        
        results = {}

        for name, model in self.models.items():

            result = np.exp(np.sqrt((cross_val_score(model, train_df, y_log, scoring = 'neg_mean_squared_error', cv = kf) * -1)))
            
            results[name] = result

            print(f'\n{name}')
            print(f'Results array : {result}')
            print(f'Result mean : {np.mean(result)}')
            print(f'Result std : {np.std(result)}')

        return self.models, results

class Ensemble_constructor():

    def __init__(self, models, structure):
        self.models = models
        self.structure = structure
        
        return

    def predict(self, test_df, ):

        if self.structure == 'weighted':
            test_predictions = (
                np.exp(self.models['Bayesian Ridge'].predict(test_df)) * 0.5 + 
                np.exp(self.models['CatBoost'].predict(test_df)) * 0.5
                )

        elif self.structure == 'single':
            test_predictions =  np.exp(self.models['Bayesian Ridge'].predict(test_df))

        return test_predictions


