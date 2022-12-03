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