import pandas as pd

def split_merge_datasets(train_df, test_df):
    '''
    Split target from training dataset.

    also

    Merge Test and Train data for handling missing values, feature transformation  / scaling.
    '''

    y = train_df[['SalePrice']]

    test_ids = test_df['Id']

    train_df =  train_df.drop(columns = ['Id', 'SalePrice'])
    test_df = test_df.drop(columns = ['Id'])

    merged_train_df = pd.concat([train_df, test_df], axis = 0)

    return y, merged_train_df, test_ids