from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    return y_pred_stat, fig

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