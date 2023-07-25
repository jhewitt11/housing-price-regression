

def fill_na_num(df):
    df = df.copy()
    for col in df.columns:
        df.loc[:, col] = df.loc[:, col].fillna(df.loc[:, col].mean())

    return df


def fill_na_cat(df):
    df = df.copy()

    columns_NA_meaningful = {
        'Alley' : 1,
        'BsmtQual' : 1,
        'BsmtCond' : 1,
        'BsmtExposure' : 1,
        'BsmtFinType1' : 1,
        'BsmtFinType2' : 1,
        'FireplaceQu' : 1,
        'GarageType' : 1,
        'GarageFinish' : 1,
        'GarageQual' : 1,
        'GarageCond' : 1,
        'PoolQC' : 1,
        'Fence' : 1,
        'MiscFeature' : 1   
    }

    for col in df.columns :

        if columns_NA_meaningful.get(col):
            df.loc[:, col] = df.loc[:, col].fillna('None')

        else:
            df.loc[:, col] = df.loc[:, col].fillna(df.loc[:, col].mode()[0])

    return df