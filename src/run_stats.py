
import pandas as pd
import numpy as np
import scipy.stats as st

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