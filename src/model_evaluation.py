from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
import scipy.stats as st

def evaluate_with_cv(df, x_columns, model, model_name='regressor'):
    """
    Model, fit, and evaluate machine learning model compared with statistical linear regression.

    """

    X = df[x_columns]
    y = df['Load-1RM-1']
    y_pred_stat = df['slope'] * df['100%MV'] + df['intercept']
    cv_results = cross_validate(model, X, y, cv=10,
        scoring=['r2', 'neg_mean_absolute_error']
        )
    scores = dict()
    scores['mae']= abs(cv_results['test_neg_mean_absolute_error'].mean())
    scores['r2'] = cv_results['test_r2'].mean()
    
    return scores

def compare_means(d1, d2, type='paired'):
    """
    Perform 2-sample t-tests and calculate Cohen's d effect size between two samples.
    """
    stats = pd.Series(dtype='float64')
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    if type=='paired': 
        # Calculate standard deviation of first sample
        s = np.sqrt(s1)
        # Calculate t-test
        stats['t statistic'], stats['ttest pvalue'] = st.ttest_rel(d1, d2)
    else:
        # calculate the pooled standard deviation
        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        # Calculate t-test
        stats['t statistic'], stats['ttest pvalue'] = st.ttest_ind(d1, d2)
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    stats['Cohens d'] = round(((u2 - u1) / s), 2)
    return stats

def cv_mae_r2(df, estimator, x_columns, cv_folds=10):
    """2022-12-02 from 23:09 2022-12-01 iteration 3 notebook.

    Determine mean absolute error through cross-validation.

    Parameters:
        - df: DataFrame containing the data.
        - estimator: Instantiate a regressor, e.g. LinearRegression(). 
        - x_columns: Feature columns in df.
        - cv_folds: Number of cross validation folds.

    Returns:
        mae, r2 (array): Results from all cross validation folds.
    """
    X = df[x_columns]
    y = df['Load-1RM-1']
    cv_results = cross_validate(estimator, X, y, cv=cv_folds,
        scoring=['neg_mean_absolute_error', 'r2']
        )
    mae = abs(cv_results['test_neg_mean_absolute_error'])
    r2 = cv_results['test_r2']
    return mae, r2

def batch_run_cv(model_names, df_dict, estimator, x_columns=['slope', 'intercept'], cv_folds=10):
    """2022-12-02 from 23:09 2022-12-01 iteration 3 notebook.
    
    Determine mean absolute error through cross-validation on multiple models.

    Parameters:
        - model_names (list): List of model names to iterate over.
        - df_dict (dict): Dictionary of DataFrames containing the data for modelling.
        - x_columns (list): List of feature names in the dataframes.
        - estimator: Instantiate a regressor, e.g. LinearRegression(). 
        - x_columns: Feature columns in df.
        - cv_folds: Number of cross validation folds.

    Returns:
        mae, r2: DataFrames containing the MAE and r^2 scores from each fold (1 column per model).
    """
    
    cv_mae = pd.DataFrame()
    cv_r2 = pd.DataFrame()
    for model in model_names:
        cv_mae[model], cv_r2[model] = cv_mae_r2(df_dict[model], estimator, x_columns, cv_folds)
    return cv_mae, cv_r2

def run_eda(df, columns=['Load-1RM-1','100%MV', 'slope', 'intercept'], mode=None,
    filepath=None):
    """2022-12-03 0:04 - See `2022-12-02 iteration 4` notebook

    Create a pairplot.
    Perform shapiro-wilks test.
    """
    stats = run_stats(df, columns=columns, mode=mode)
    pairplot = sns.pairplot(df[columns], diag_kind='kde', corner=True)
    if filepath:
        pairplot.savefig(f'{filepath} pairplot.png')
    return stats