from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
import scipy.stats as st
import numpy as np

import sys
sys.path.append(r"C:\Users\silvh\OneDrive\lighthouse\custom_python")
from silvhua import *


def evaluate_regression(y_test, y_pred, y_train, y_pred_train, model_name='regressor',plot=True):
    """2022-11-27 21:23
    * Print model evalutation metrics: 
        * RMSE
        * Mean absolute error (MAE)
        * Mean error
        * R^2 score
        * Pearson correlation coefficient
    * If plot=True : Provide scatterplot of true vs. predicted values.
    Params:
        - y_test, y_pred (array): True and predicted values for test set.
        - y_train, y_pred_train (array): True and predicted values for train set.

        Optional:
        - model_name (str): Name of the model to print and to add to the figure title.
        - plot (bool): If true, plot true vs. predicted values using test data set from train-test split.

    Returns: 
        - If plot=True, returns a figure of the scatterplot.
    """
    # Metrics for test data
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mean_abs_error = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mean_error = (y_pred-y_test).mean()

    # Metrics for training data

    rmse_train = mean_squared_error(y_train, y_pred_train)
    mean_abs_error_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    mean_error_train = (y_pred_train-y_train).mean()
    
    # Calculate Pearson Correlation between predicted and true values:
    pearson = stats.pearsonr(y_test, y_pred)
    pearson_train = stats.pearsonr(y_train, y_pred_train)

    print(f'\n{model_name} evaluation metrics: \n\t\tTest data\tTraining data\t\tDifference')
    print(f'RMSE: \t\t\t{rmse:.2f}\t\t{rmse_train:.2f}\t\t{(rmse - rmse_train):.2f}')
    print(f'MAE: \t\t\t{mean_abs_error:.2f}\t\t{mean_abs_error_train:.2f}\t\t{(mean_abs_error - mean_abs_error_train):.2f}')
    print(f'mean error: \t\t{mean_error:.2f}\t\t{mean_error_train:.2f}\t\t{(mean_error-mean_error_train):.2f}')
    print(f'R^2: \t\t\t{r2:.2f}\t\t{r2_train:.2f}\t\t{(r2 - r2_train):.2f}')
    print(f'Pearson r statistic: \t{pearson.statistic:.2f}\t\t{pearson_train.statistic:.2f}\t\t{pearson.statistic-pearson_train.statistic:.2f}')
    print(f'\t\t\tp={pearson.pvalue:.2f}\t\tp={pearson_train.pvalue:.2f}')
    print(f'\npredictions mean: \t{y_pred.mean():.2f}\t\t{y_pred_train.mean():.2f}\t\t{(y_pred.mean() - y_pred_train.mean()):.2f}')
    print(f'predictions std: \t{y_pred.std():.2f}\t\t{y_pred_train.std():.2f}\t\t{(y_pred.std() - y_pred_train.std()):.2f}')
    print(f'\ntarget mean: \t\t{y_test.mean():.2f}\t\t{y_train.mean():.2f}\t\t{y_test.mean()-y_train.mean():.2f}')
    print(f'target std: \t\t{y_test.std():.2f}\t\t{y_train.std():.2f}\t\t{y_test.std()-y_train.std():.2f}')

    if plot:
        ax = sns.lineplot(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], ls=':', alpha=0.5)
        fig = sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, ax=ax)
        fig.set_xlabel('Measured')
        fig.set_ylabel('Predicted')
        fig.set_title(model_name)
        return fig

def model_data(df_train, df_test, x_columns, model, model_name='regressor'):
    """2022-11-27 21:31
    Train and evaluate model using a train-test split.

    Parameters:
        - df_train, df_test: DataFrames with train and test data (1 row per participant).
        - x_columns (list or str): Column names of features to be used for modelling.
        - model: Instance of an estimator.
        - model_name (str, optional): Name of the model for printed results and figure title.

    Returns:
        - y_pred_train, y_pred (array): Model predictions for the train and test sets, respectively.
        - features (dict): Model features and their coefficients.
        - Figure of the scatterplot.

    Syntax:
        y_pred_train, y_pred, features, figure = model_data(df_train, df_test, 
            x_columns=['slope', 'intercept'], model=model, model_name='Model')
    """
    X_train = df_train[x_columns]
    X_test = df_test[x_columns]
    y_train = df_train['Load-1RM-1']
    y_test = df_test['Load-1RM-1']
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    fig = evaluate_regression(y_test, y_pred, y_train, y_pred_train, model_name=model_name, plot=True)
    
    print('\nModel feature coefficients')
    features = dict()
    for index, value in enumerate(model.feature_names_in_):
        features['LV '+value] = model.coef_[index]
        print(f'\tLV {value}: {model.coef_[index]}')
    features['model intercept'] = model.intercept_
    print('\tmodel intercept: ', model.intercept_)

    return y_pred_train, y_pred, features, fig

def compare_ml_stat(y_test, y_pred, y_pred_stat, model_name='regressor',plot=True):
    """2022-11-27 22:23
    * Print model evalutation metrics for a given model and for the baseline statistical
    regression model (for comparison): 
        * RMSE
        * Mean absolute error (MAE)
        * Mean error
        * R^2 score
        * Pearson correlation coefficient
    * If plot=True : Provide scatterplot of true vs. predicted values.
    Params:
        - y_test, y_pred (array): True and predicted values for test set.
        - y_pred_stat (array): Predicted values based on statistical linear regression.

        Optional:
        - model_name (str): Name of the model to print and to add to the figure title.
        - plot (bool): If true, plot true vs. predicted values using test data set from train-test split.

    Returns: 
        - Figure: Scatterplot of true vs. predicted values.
    """
    # Metrics for test data
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mean_abs_error = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mean_error = (y_pred-y_test).mean()
    
    # Metrics for statistical linear regression
    rmse_stat = mean_squared_error(y_test, y_pred_stat, squared=False)
    mean_abs_error_stat = mean_absolute_error(y_test, y_pred_stat)
    r2_stat = r2_score(y_test, y_pred_stat)
    mean_error_stat = (y_pred_stat-y_test).mean()

    # Calculate Pearson Correlation between predicted and true values:
    pearson = stats.pearsonr(y_test, y_pred)
    pearson_stat = stats.pearsonr(y_test, y_pred_stat)
    
    print(f'\n{model_name} evaluation metrics: \n\t\tModel of interest\tBaseline\tDifference')
    print(f'RMSE: \t\t\t{rmse:.2f}\t\t{rmse_stat:.2f}\t\t{(rmse - rmse_stat):.2f}')
    print(f'MAE: \t\t\t{mean_abs_error:.2f}\t\t{mean_abs_error_stat:.2f}\t\t{(mean_abs_error - mean_abs_error_stat):.2f}')
    print(f'mean error: \t\t{mean_error:.2f}\t\t{mean_error_stat:.2f}\t\t{(mean_error-mean_error_stat):.2f}')
    print(f'R^2: \t\t\t{r2:.2f}\t\t{r2_stat:.2f}\t\t{(r2 - r2_stat):.2f}')
    print(f'Pearson r statistic: \t{pearson.statistic:.2f}\t\t{pearson_stat.statistic:.2f}\t\t{pearson.statistic-pearson_stat.statistic:.2f}')
    print(f'\t\t\tp={pearson.pvalue:.2f}\t\tp={pearson_stat.pvalue:.2f}')
    print(f'\npredictions mean: \t{y_pred.mean():.2f}\t\t{y_pred_stat.mean():.2f}\t\t{(y_pred.mean() - y_pred_stat.mean()):.2f}')
    print(f'predictions std: \t{y_pred.std():.2f}\t\t{y_pred_stat.std():.2f}\t\t{(y_pred.std() - y_pred_stat.std()):.2f}')
    print(f'\ntarget mean: {y_test.mean():.2f}')
    print(f'target std: {y_test.std():.2f}')

    if plot:
        ax = sns.lineplot(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], ls=':', alpha=0.5)
        fig = sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, ax=ax)
        fig.set_xlabel('Measured')
        fig.set_ylabel('Predicted')
        fig.set_title(model_name)
        return fig

def stat_modelling(df, model_name='regressor', mvt='individual'):
    """2022-11-27 22:31

    Run statistical modelling using the formula 1RM = (LV_slope) * MVT + LV_intercept.
    Compare results with that obtained from statistical model using individual MVT.

    Parameters:
        - df: DataFrame.
        - model_name (str): Model name to display when printing results and for figure title.
        - mvt (str): {'individual' or 'group'}) Whether to use individual MVT or 
            group mean MVT in the statistical modelling.

    Returns:
        - y_pred (array): Model predictions.
        - Figure: Scatterplot of true vs. predicted values.

    Syntax:
        y_pred, fig = stat_modelling(sm_test, model_name='Statistical linear regression', 
            mvt='individual')
    """
    y_pred_stat = df['slope'] * df['100%MV'] + df['intercept']
    if mvt=='individual':
        y_pred = y_pred_stat
    elif mvt=='group':
        y_pred = df['slope'] * df['group MVT'] + df['intercept']
    y_test = df['Load-1RM-1']
    fig = compare_ml_stat(y_test, y_pred, y_pred_stat, model_name=model_name, plot=True)
    return y_pred, fig

def model_data_vs_stat(df, x_columns, model, model_name='regressor'):
    """
    Model, fit, and evaluate machine learning model compared with statistical linear regression.

    Parameters:
        - df_train: DataFrame with 1 row per participant.
        - x_columns (list or str): Column names of features to be used for modelling.
        - model: Instance of an estimator.
        - model_name (str, optional): Name of the model for printed results and figure title.

    Returns:
        - y_pred (array): Model predictions for the estimator.
        - features (dict): Model features and their coefficients.
        - Figure of the scatterplot.

    Syntax:
        y_pred, figure, features = model_data_vs_stat(df, 
            x_columns=['slope', 'intercept'], model=model, model_name='Model')
    """

    X_test = df[x_columns]
    y_test = df['Load-1RM-1']
    y_pred_stat = df['slope'] * df['100%MV'] + df['intercept']
    model.fit(X_test, y_test)
    y_pred = model.predict(X_test)

    fig = compare_ml_stat(y_test, y_pred, y_pred_stat, model_name=model_name,plot=True)
    print('\nModel feature coefficients')
    features = dict()
    for index, value in enumerate(model.feature_names_in_):
        features['LV '+value] = model.coef_[index]
        print(f'\tLV {value}: {model.coef_[index]}')
    features['model intercept'] = model.intercept_
    print('\tmodel intercept: ', model.intercept_)

    return y_pred, fig, features

def run_all_models(stat_models_dict, ml_models_dict, df, x_columns=['slope', 'intercept'], pickle_name=None,
    path=r'C:\Users\silvh\OneDrive\lighthouse\projects\lighthouse-capstone-project\output'):
    """2022-12-02 23:59 See `2022-12-02 iteration 4` notebook

    Run models 1-4 for single dataframe.
    """
    # Initialize dataframes for storing model outputs
    predictions = pd.DataFrame()
    coefficients = pd.DataFrame()
    cv_metrics = pd.DataFrame()
    stats = pd.DataFrame()
    
    # Add true y value
    predictions['Measured'] = df['Load-1RM-1']

    for model, mvt_param in stat_models_dict.items():
        predictions[model], fig = stat_modelling(df, model, mvt=mvt_param)
    for model, model_instance in ml_models_dict.items():
        predictions[model], fig, coefficients[model] = model_data_vs_stat(
            df, x_columns, model=model_instance, model_name=model
        )
        # cross-validation metrics
        cv_metrics[model] = evaluate_with_cv(df, x_columns, 
            model=model_instance, model_name=model)

        # ttest and Cohen's d effect size
        stats[model] = compare_means(
                df['Load-1RM-1'], # True y value
                predictions[model], type='paired') # Model predicts

        # Concatenate
        metrics = pd.concat([cv_metrics, 
            stats, 
            coefficients
            ], axis=0)
        metrics = round(metrics, 4)
        # pickle the model
        if pickle_name:
            savepickle(model_instance, f'{pickle_name} {model}', path=path+'\models')
    # save predictions and metrics
    if pickle_name:
        save_csv(predictions, f'{pickle_name} predictions', path=path+'\predictions')
        save_csv(metrics, f'{pickle_name} metrics and coefficients', path=path+'\models')

    return predictions, metrics, ml_models_dict

def batch_model(model_names, df_dict, estimator=None, x_columns=['slope', 'intercept'], pickle_name=None,
    path=r'C:\Users\silvh\OneDrive\lighthouse\projects\lighthouse-capstone-project\output'):
    """2022-12-02 23:06
    Fit and evaluate multiple dataframes using given estimator. Meant for testing multiple 
        feature engineering iterations.
    Parameters:
        - model_names (list): List of model names to iterate over.
        - df_dict (dict): Dictionary of DataFrames containing the data for modelling.
        - x_columns (list): List of feature names in the dataframes.
        - pickle_name (str): Root of filename for saving results. If None, results are not automatically saved.

    Returns:
        - predictions (DataFrame): Predictions from each of the models.
        - metrics (DataFrame): Cross validation metrics, model coefficients, 
            paired ttest results and Cohen's d effect size between predicted vs. true results.
        - model_dict (dict): Dictionary containing the trained models. 

    Example syntax:
        predictions, metrics, model_dict = batch_model(model_names, df_fw_dict, pickle_name='03 iteration')
    """
    # Initialize dataframes for storing model outputs
    predictions = pd.DataFrame()
    coefficients = pd.DataFrame()
    cv_metrics = pd.DataFrame()
    stats = pd.DataFrame()
    model_dict = dict()
    
    # Add true y value
    predictions['Measured'] = df_dict[model_names[0]]['Load-1RM-1']

    for model in model_names:
        # Run model
        if (estimator==None):
            model_dict[model] = LinearRegression() 
            predictions[model], fig, coefficients[model] = model_data_vs_stat(
                df_dict[model], x_columns, model=model_dict[model], model_name=model
            )
        else: 
            model_dict[model] = estimator
            model_dict[model].fit(df_dict[model][x_columns], df_dict[model]['Load-1RM-1'])
            predictions[model] = model_dict[model].predict(df_dict[model][x_columns])
        # cross-validation metrics
        cv_metrics[model] = evaluate_with_cv(df_dict[model], x_columns, 
            model=model_dict[model], model_name=model)

        # ttest and Cohen's d effect size
        stats[model] = compare_means(
                df_dict[model]['Load-1RM-1'], # True y value
                predictions[model], type='paired') # Model predicts

        # Concatenate
        metrics = pd.concat([cv_metrics, 
            stats, 
            coefficients
            ], axis=0)
        metrics = round(metrics, 4)
        # pickle the model
        if pickle_name:
            savepickle(model_dict[model], f'{pickle_name} {model}', path=path+'\models')
        
    # save predictions and metrics
    if pickle_name:
        save_csv(predictions, f'{pickle_name} predictions', path=path+'\predictions')
        save_csv(metrics, f'{pickle_name} metrics and coefficients', path=path+'\models')

    return predictions, metrics, model_dict

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