import pandas as pd
import numpy as np

from pipe import Preprocessor
from pipe import Trainer
from pipe import Ensemble_constructor


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


''' Handles cleaning & preprocessing steps '''
preprocessor = Preprocessor()
train_df, y_log, test_df, test_ids = preprocessor.run(train_df, test_df)


''' Returns trained models and their results from KFold cross validation '''
trainer = Trainer()
models, results = trainer.run(train_df, y_log)


ensemble = Ensemble_constructor(models = models, structure = 'weighted')
preds = ensemble.predict(test_df)


''' Format for submission '''
preds_df = pd.DataFrame({'Id' : test_ids.values, 'SalePrice' : preds})
preds_df.to_csv('results/submission.csv', index = False)