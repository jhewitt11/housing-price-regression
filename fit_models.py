
import numpy as np
import pandas as pd

from sklearn.linear_model import BayesianRidge, OrthogonalMatchingPursuit, Ridge
from sklearn.model_selection import KFold, cross_val_score


models = {
    'Bayesian Ridge' : BayesianRidge(),
    'Ridge Regression' : Ridge(),
    'Orthogonal Matching Pursuit' : OrthogonalMatchingPursuit(),
}

data = pd.read_csv('train_prepped.csv')
test_x = pd.read_csv('test_prepped.csv')

log_y = data['SalePrice']
data = data.drop(['SalePrice'], axis = 1)




for name, model in models.items():
    model.fit(data, log_y)
    print(f'{name} trained.')


kf = KFold(n_splits = 10)

results = {}

for name, model in models.items():

    result = np.exp(np.sqrt((cross_val_score(model, data, log_y, scoring = 'neg_mean_squared_error', cv = kf) * -1)))

    results[name] = result

    print(f'\n{name}')
    print(f'Results array : {result}')
    print(f'Result mean : {np.mean(result)}')
    print(f'Result std : {np.std(result)}')


test_predictions = (
    np.exp(models['Bayesian Ridge'].predict(test_x)) * 0.5 + 
    np.exp(models['Ridge Regression'].predict(test_x)) * 0.25 +
    np.exp(models['Orthogonal Matching Pursuit'].predict(test_x)) * 0.25
    )


Ids = list(range(1461, 1461 + len(test_x)))

submission = pd.DataFrame({'Id' : Ids,'SalePrice' : test_predictions}, )
submission.to_csv('results/submission.csv', index = False)

# Root Mean Squared Log Error
# 00: 0.12599