import numpy as np
from scipy.stats import skew

def unskew_numeric(df):

    df = df.copy()

    for col in df.columns:

        skewness = np.abs(skew(df[col].values))
        
        if skewness >= 0.5 :
            df.loc[:, col] = np.log1p(df.loc[:, col])


    return df