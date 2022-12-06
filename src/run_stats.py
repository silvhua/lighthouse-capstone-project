
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