import pandas as pd
from sklearn.linear_model import LinearRegression
import re
import scipy.stats as st

def linear_regression(df):
    """
    Calculate slope and intercept using linear regression, where X = load, y = velocity.
    Function called by other functions: individual_regression() and reshape_group_df_lr(df).
    Can be used as a stand-alone function or when called with .apply with transposed DataFrame.

    Parameters:
        df: DataFrame with each row containing data for an individual.
    Returns:
        Dataframe with the slope and intercept for the linear regression.
    
    2022-11-27 20:32
    """
    from sklearn.linear_model import LinearRegression

    if type(df)==pd.Series:
        velocity_columns = df.index[df.index.str.contains('MV')]
        load_columns = df.index[df.index.str.contains('Load')]

        load = df[load_columns].values.reshape(-1,1)
        velocity = df[velocity_columns].values.reshape(-1,1)

        lr = LinearRegression()
        lr.fit(velocity, load)
        
        # LinearRegression attributes are in arrays, so need to access values with indices
        df['slope'] = lr.coef_[0][0] 
        df['intercept'] = lr.intercept_[0]

        return df
    else:
        velocity_columns = df.columns[df.columns.str.contains('MV')]
        load_columns = df.columns[df.columns.str.contains('Load')]
        load = df[load_columns].values.reshape(-1,1)
        velocity = df[velocity_columns].values.reshape(-1,1)

        lr = LinearRegression()
        lr.fit(velocity, load)

        df_lr = pd.DataFrame()
        df_lr['slope'] = lr.coef_[0]
        df_lr['intercept'] = lr.intercept_

        return df_lr

def linear_regression2(df, loads):
    """2022-11-28 18:22
    Calculate slope and intercept using linear regression, where X = load, y = velocity.
    Function called by other functions: individual_regression2() and reshape_group_df_lr(df).
    Can be used as a stand-alone function or when called with .apply with transposed DataFrame.

    Parameters:
        - df: DataFrame with each row containing data for an individual.
        - loads (list): List of relative loads to be used for calculating LV slope and LV intercept.
    Returns:
        Dataframe with the slope and intercept for the linear regression.

    """
    from sklearn.linear_model import LinearRegression

    # Select columns that contain the numbers in load
    regex_MV = ''.join([str(load)+r'%.*MV|' for load in loads])
    regex_load = ''.join([str(load)+r'%1RM|' for load in loads]) 
    if type(df)==pd.Series:
        load = df.filter(regex=regex_load[:-1], axis='index').values.reshape(-1,1)
        velocity = df.filter(regex=regex_MV[:-1], axis='index').values.reshape(-1,1)

        lr = LinearRegression()
        lr.fit(velocity, load)
        
        # LinearRegression attributes are in arrays, so need to access values with indices
        df['slope'] = lr.coef_[0][0] 
        df['intercept'] = lr.intercept_[0]

        return df
    else:
        load = df.filter(regex=regex_load[:-1]).values.reshape(-1,1)
        velocity = df.filter(regex=regex_MV[:-1]).values.reshape(-1,1)
        lr = LinearRegression()
        lr.fit(velocity, load)

        df_lr = pd.DataFrame()
        df_lr['slope'] = lr.coef_[0]
        df_lr['intercept'] = lr.intercept_

        return df_lr
        
def individual_regression2(df, loads):
    """2022-11-28 20:09

    Necessary for feature engineering.
    Calculate slope and intercept for each row of the dataframe (i.e. for each individual participant)
    by calling the linear_regression2 function.

    Parameters:
        - df: DataFrame with each row containing data for a single individual.
        - loads (list): List of relative loads to be used for calculating LV slope and LV intercept.
    Returns:
        Dataframe with new columns added: 
            - 'slope' and 'intercept' for the linear regression for each individual row.
            - 'group MVT': Mean '100%MV' value for the dataset (value identical in each row)
    """
    df_lr = df.transpose().apply(lambda x:linear_regression2(x, loads)).transpose()
    
    df_lr['group MVT'] =  df_lr['100%MV'].mean()
    print('Dataframe shape: ', df_lr.shape)
    return df_lr

def sorted_test_split(df_fw, df_sm):
    """2022-11-27 20:34
    Sort participants by free weight squat 1RM, then perform train-test split so 
    train and test groups have similar free weight 1RM values.

        Parameters:
            - df_fw: Dataframe containing free weight data set.
            - df_sm: DataFrame containing smith machine data set.
        Returns:
            - fw_train, fw_test, sm_train, sm_test : 4 dataframes containing train and test sets.
        
        Syntax:
        fw_train, fw_test, sm_train, sm_test = sorted_test_split(df_fw, df_sm)
    """
    test_sorted_df_implicit_index = [i for i in range(3,len(df_sm),5)]
    print(f'Original df shapes: {df_fw.shape}, {df_sm.shape}')
    fw_test = df_fw.sort_values('Load-1RM-1').iloc[test_sorted_df_implicit_index, :]
    test_exp_index = fw_test.index
    train_index = df_fw.index[~df_fw.index.isin(test_exp_index)]
    fw_train = df_fw.loc[train_index, :]
    
    # ensure train-test split has same participants for smith machine data as in free weight data
    sm_test = df_sm.loc[test_exp_index, :] 
    sm_train = df_sm.loc[train_index, :]
    print(f'Test index: {test_exp_index.to_list()}')
    print(f'Train index: {train_index.to_list()}')
    print(f'Train shapes: {fw_train.shape}, {sm_train.shape}')
    print(f'Test shapes: {fw_test.shape}, {sm_test.shape}')
    return fw_train, fw_test, sm_train, sm_test

def create_pairs(sortable_list):
    """
    Create all unique pair combinations of 2 elements from a list (exclude pairs where
    the elements are identical)

    Parameters:
        - list (list): List of numbers or strings (or other elements that can be passed into
            the sorted() function).
    Returns:
        List of pair combinations.
    Syntax:
        loads = [20, 40, 60, 80, 90]
        unique_load_pairs = create_pairs(loads)
    """
    from itertools import product
    pairs = list(product(sortable_list, sortable_list))
    pairs = [set(sorted([pair[0], pair[1]])) for pair in pairs if pair[0] != pair[1]]
    unique_pairs = []
    for item in pairs:
        if item not in unique_pairs:
            unique_pairs.append(item)
    unique_pairs = [sorted(list(pair)) for pair in unique_pairs]
    print(f'Number of unique pairs: {len(unique_pairs)}')
    return unique_pairs

def run_stats(df, columns=['slope', 'intercept', 'Load-1RM-1'], mode=None):
    """
    Perform Shapiro-Wilk test for normality.

    Parameters:
        - df (DataFrame)
        - columns (list): Column names on which to perform Shapiro-Wilk test
        - mode ('binary' or None): If binary, return 1 if null hypothesis rejected (not normal).
            Default is to return p-value.
    Returns: Series with results from the Shapiro-Wilk test for normality. 
        - Default results are the p-value. If mode='binary', values are 0 or 1.
    """
    alpha = 0.05
    statistics = pd.Series(dtype='float64')
    for column in columns:
        if mode == 'binary':
            statistics['normal '+column] = 1 if st.shapiro(df[column]) < alpha else 0
        else:
            statistics['normal '+column] = round(st.shapiro(df[column])[1], 3)

    return statistics