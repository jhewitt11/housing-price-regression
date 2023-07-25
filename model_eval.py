import numpy as np
import pandas as pd
import random

from pycaret.regression import *

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


from pipe import Preprocessor
from pipe import Trainer
from pipe import Ensemble_constructor




train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

''' Handles cleaning & preprocessing steps '''
preprocessor = Preprocessor()
train_df, y_log, test_df, test_ids = preprocessor.run(train_df, test_df)


data = pd.concat([train_df, y_log], axis = 1)

s = setup(data, target = 'SalePrice', session_id = 000)

best = compare_models()

df = pull()

ID = random.randint(1000, 9999)
df.to_csv(f'pycaret_results/model_results_{str(ID)}.csv')







