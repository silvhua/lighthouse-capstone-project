
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LinearRegression
import numpy as np
import re
from processing_functions import *

def reshape_group_df_lr2(df, loads):
    """2022-11-27 20:39
    Necessary for data visualization.
    Reshape dataframe each row represents data from one rep (each participant has multiple columns).
    Add the estimations from the linear regression.

        Parameters:
        - df: DataFrame with one row per participant. 
        - loads (list): List of relative loads to be used for calculating LV slope and LV intercept.
        Returns:
        - DataFrame with each row representing a single set (each participant may have multiple rows). 
        Participants are sorted by strength for compatibility with Seaborn plots.
    """
    # Calculate slope and intercept for each row of the dataframe (i.e. for each individual participant)
    # by calling the linear_regression function.
    print('Original shape: ',df.shape)
    df = df.transpose().apply(lambda x:linear_regression2(x, loads)).transpose()
    
    velocity_columns = df.columns[df.columns.str.contains('MV')]
    load_columns = df.columns[df.columns.str.contains('Load')]

    # Use intercept and slope to calculate predicted load
    for column in velocity_columns:
        df[re.sub('(\d*%).*','\\1 predicted load', column)] = df['slope'] * df[column] + df['intercept']
    
    prediction_columns = df.columns[df.columns.str.contains('predicted load')].to_list()

    # Sort participants by strength
    df = df.sort_values('Load-1RM-1').reset_index(drop=True)
    df2 = pd.concat([
        df.melt(
            value_vars=load_columns, value_name='absolute load',
            ignore_index=False
        ),
        df.melt(
            id_vars='Load-1RM-1',
            value_vars=velocity_columns, var_name='%1RM', value_name='mean velocity',
            ignore_index=False
            ),
    ], axis=1).drop(columns='variable')
    df3 = df.melt(
            # id_vars='Load-1RM-1',
            value_vars=prediction_columns, value_name='predicted load', 
            ignore_index=False
            ).reset_index(drop=True).drop(columns='variable')
    df2['%1RM'] = df2['%1RM'].str.replace('(\d*)\D*','\\1', regex=True).astype(float)
    df2 = df2.rename({'Load-1RM-1':'1RM'}, axis=1)
    df2 = df2.reset_index(names='participant')
    df2 = pd.concat([df2,df3],axis=1)
    print('New shape: ', df2.shape)
    return df2
    
def reshape_group_df_lr(df):
    """2022-11-27 20:39
    Necessary for data visualization.
    Reshape dataframe each row represents data from one rep (each participant has multiple columns).
    Add the estimations from the linear regression.

        Parameters:
        - df: DataFrame with one row per participant. Must contain these columns:
            'Load<number>%1RM'
            'Load-1RM-1', 
            '<number>% MV',
            '100%MV'
        Returns:
        - DataFrame with each row representing a single set (each participant may have multiple rows). 
        Participants are sorted by strength for compatibility with Seaborn plots.
    """
    # Calculate slope and intercept for each row of the dataframe (i.e. for each individual participant)
    # by calling the linear_regression function.
    print('Original shape: ',df.shape)
    df = df.transpose().apply(lambda x:linear_regression(x)).transpose()
    
    velocity_columns = df.columns[df.columns.str.contains('MV')]
    load_columns = df.columns[df.columns.str.contains('Load')]

    # Use intercept and slope to calculate predicted load
    for column in velocity_columns:
        df[re.sub('(\d*%).*','\\1 predicted load', column)] = df['slope'] * df[column] + df['intercept']
    
    prediction_columns = df.columns[df.columns.str.contains('predicted load')].to_list()

    # Sort participants by strength
    df = df.sort_values('Load-1RM-1').reset_index(drop=True)
    df2 = pd.concat([
        df.melt(
            value_vars=load_columns, value_name='absolute load',
            ignore_index=False
        ),
        df.melt(
            id_vars='Load-1RM-1',
            value_vars=velocity_columns, var_name='%1RM', value_name='mean velocity',
            ignore_index=False
            ),
    ], axis=1).drop(columns='variable')
    df3 = df.melt(
            # id_vars='Load-1RM-1',
            value_vars=prediction_columns, value_name='predicted load', 
            ignore_index=False
            ).reset_index(drop=True).drop(columns='variable')
    df2['%1RM'] = df2['%1RM'].str.replace('(\d*)\D*','\\1', regex=True).astype(float)
    df2 = df2.rename({'Load-1RM-1':'1RM'}, axis=1)
    df2 = df2.reset_index(names='participant')
    df2 = pd.concat([df2,df3],axis=1)
    print('New shape: ', df2.shape)
    return df2

def plot_profiles_lr(df, y='absolute load', x='mean velocity', row='participant', 
    y_pred='predicted load', show_legend=False, yaxis_label=None, title=None, scale=False):
    """2022-11-27 20:55
    Make a figure containing subplots for each individual participant, where each subplot is the 
    load-velocity profile (load on y-axis, velocity on x-axis) with the linear regression line.

    Parameters:
        * df: Dataframe that is reshaped using the reshape_group_df_lr() function.
        * title (str): Figure title. If none, will be blank.

        Optional:
            * y (str): Column name with y-axis data. Default is 'absolute load'.
            * x (str): Column name with x-axis data. Default is 'mean velocity'.
            * y_pred (str): Column name with the linear regression prediction. Default is 'predicted load'.
            * scale (bool): If True, all y-axes will have the same range.
            * legend (bool): Whether or not to show the legend in the first subplot. Default is False.
    Returns:
        Figure.

    Syntax:
    figure = plot_profiles_lr(reshape_group_df_lr(df), 
        title='Load-Velocity Profiles', show_legend=True)
    """
    
    # sns.reset_defaults()    
    # %matplotlib inline
    participants = sorted(df[row].unique())
    subplot_label = [x+1 for x in range(-1,100)]
    nrows = round((len(participants)+1)/4)

    title_variable = df[row].name
    fig, ax = plt.subplots(nrows=nrows ,ncols=4, figsize=(10,nrows*2.5))
    fig.suptitle(title, fontsize=20)
    ymin = df[y].min()
    ymax = df[y].max()
    ax = ax.flatten()
    # colors = sns.color_palette("rocket", as_cmap = True)

    ax_index = 0
    for participant in participants:
        if (show_legend==True) & (ax_index==0):
            legend = 'full'
        else:
            legend = False
        filter = (df[row] == participant)

        # Plot measured values
        sns.lineplot(data=df[filter], y=y, 
            x=x, marker='o', alpha=0.9,
            legend=legend, label='measured',
                ax = ax[ax_index])
        # Plot predicted values
        sns.lineplot(data=df[filter], y=y_pred, 
            x=x, alpha=0.9, label='regression', ls=':',
            legend=legend,
                ax = ax[ax_index])
        if nrows > 1:
            ax[ax_index].set_title(f'{subplot_label[ax_index]}) {title_variable} {participant}', loc='left')
        else:
            ax[ax_index].set_title(f'{subplot_label[ax_index]})', loc='left')
        if scale==True:
            ax[ax_index].set_ylim([ymin,ymax]) # Make the y axes all the same
        if yaxis_label:
            ax[ax_index].set_ylabel(yaxis_label)
        if (legend == 'full'):
            ax[ax_index].legend()

        ax_index += 1
        
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

def data_viz(df_fw, df_sm):
    """2022-11-27 21:21
    Plot data from free weight and Smith machine data sets for all participants along with 
    group minimum velocity threshold (MVT).
    Top row of subplots show load-velocity profiles using absolute load.
    Bottom row of subplots show load-velocity profiles using relative load.
    Requires the custom function reshape_group_df_lr().

    Parameters:
        df_fw, df_sm: Original DataFrames with free weight and Smith machine data, respectively
        (1 participant per row).

    Returns: 
        Figure

    Syntax: 
    data_viz_fig = data_viz(df_fw, df_sm)
    """
    # sns.reset_defaults()    
    # %matplotlib inline
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
    xmin = pd.concat([df_fw['100%MV'],df_sm['100%MV']], axis=0).min()
    xmax = pd.concat([df_fw['20% MV'],df_sm['20% MV']], axis=0).max()
    # print(xmin, xmax)
    
    for index, df in enumerate([df_fw, df_sm]):
        # absolute load
        sns.lineplot(data=reshape_group_df_lr(df), x='mean velocity', y='absolute load', 
            hue='participant', alpha=.8,marker='.', size=1, ls=':',
            legend=False, ax=ax[0, index]
        ).set_title('Smith machine' if index==1 else 'Free weight')
        ax[0, index].axvline(x=df_fw['100%MV'].mean(),
            label='group MVT',ls='--',alpha=.5)
        ax[0, index].set_xlim([xmin,xmax])

        # relative load
        sns.lineplot(data=reshape_group_df_lr(df), x='mean velocity', y='%1RM', 
            hue='participant', alpha=.8, marker='.', size=1, ls=':',
            legend=False, ax=ax[1, index]
        )
        ax[1, index].axvline(x=df_fw['100%MV'].mean(),
            label='group MVT',ls='--',alpha=.5)
        ax[1, index].set_xlim([xmin,xmax])

    ax[0,0].legend()
    return fig

def compare_models(fw_predictions, sm_predictions, title='Measured 1RM vs. model predictions', 
    context='talk', annotate=True, ymin=-1.5, ymax=5):
    """2022-11-27 23:00

    Plot predictions from all the models for each of the free weight and smith machine data sets.
        Parmaters:
            - fw_predictions, sm_predictions (DataFrame): 
                Dataframes that each contain target data ('Measured' column) and model predictions
                (1 column per model).
            - title (str): Overall plot title.
            - context (None or str): Seaborn .set_theme() parameter. 
                One of {paper, notebook, talk (default), poster}. If None, set to 'default (notebook)'.
            - annotate (bool): Whether or not to annotate the bar graph with values. Default is True.
        Returns:
            - figure with scatter plots of measured vs. predicted values for all models.
            - figure with bar charts of mean absolute error and mean error for all models.
            - DataFrames for each of the free weight and smith machine data sets containing:
                mean absolute error and mean error for all models.

    Command syntax:
        scatterplot, error_plot, fw_metrics, sm_metrics = compare_models(fw_predictions, 
        sm_predictions, title='Model predictions', context='talk')

        scatterplot.savefig('../output/figures/Measured vs predicted for all samples.png')
        error_plot.savefig('../output/figures/Error bar chart for all samples.png')
        path = r'../output/predictions'
        save_csv(fw_metrics, 'free weight errors', path=path)
        save_csv(sm_metrics, 'smith machine errors', path=path)
    """
    fw_models = fw_predictions.columns[1:].to_list()
    # sns.reset_defaults()    
    # %matplotlib inline
    font_scale=.8 if context=='talk' else 1
    rc={'lines.markersize': 6} if context=='talk' else None
    sns.set_theme(context=context, style='ticks', font_scale=font_scale, 
        rc=rc)
    ncols = len(fw_models)
    fig, ax = plt.subplots(nrows=2, ncols=ncols, figsize=(3*ncols, 2*3))
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(3*ncols, 5))
    fw_error = pd.DataFrame()
    sm_error = pd.DataFrame()
    for index, model in enumerate(fw_models):
        # Calculate error
        fw_error[model] = fw_predictions[model] - fw_predictions['Measured'] 
        sm_error[model] = sm_predictions[model] - sm_predictions['Measured'] 

        # Plot measured vs. predicted values for each model
        ax[0, index].axline(xy1=(150, 150), slope=1, alpha=0.8, linewidth=0.5, color='orange')
        if index == 0:
            ax[0, index].set_ylabel('Predicted') 
        sns.scatterplot(data=fw_predictions, x='Measured', y=model, ax=ax[0, index],
            alpha=0.5, marker='o', legend=(True if index==0 else False), label='FW',
        ).set(xlabel=None, ylabel=None)
        ax[0, index].set_title(model)
        if index == 0:
            ax[0, index].set_ylabel('Predicted') 

        ax[1, index].axline(xy1=(150, 150), slope=1, alpha=0.8, linewidth=0.5, color='orange')
        sns.scatterplot(data=sm_predictions, x='Measured', y=model, ax=ax[1, index],
            alpha=0.5, marker='s', legend=(True if index==0 else False), 
            label='SM', color='slateblue',
        ).set(ylabel=None)
        if index == 0:
            ax[1, index].set_ylabel('Predicted')   
        
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig2.tight_layout(rect=[0, 0, 1, 0.9])

    # Calculate remaining evaluation metrics and reshape dataframe for plotting
    fw_error['Metric'] = 'Error'
    fw_mae = abs(fw_error.iloc[:,:-1])
    fw_mae['Metric'] = 'MAE'
    fw_metrics = pd.concat([fw_error, fw_mae], axis=0).melt(
        value_vars=fw_models, id_vars=['Metric'], var_name='model')
    print(f'Metrics dataframe shape (free weight data): {fw_metrics.shape}')

    sm_error['Metric'] = 'Error'
    sm_mae = abs(sm_error.iloc[:,:-1])
    sm_mae['Metric'] = 'MAE'
    sm_metrics = pd.concat([sm_error, sm_mae], axis=0).melt(
        value_vars=fw_models, id_vars=['Metric'], var_name='model')
    print(f'Metrics dataframe shape (Smith machine data): {sm_metrics.shape}')
    
    # Plot evaluation metrics: 
    sns.barplot(data=fw_metrics, y='value', x='model', hue='Metric', 
        errorbar=('se', 1.96), # error bars set to 95% confidence interval, or 1.96*standard error
        ax=ax2[0]).set_ylim([ymin, ymax])
    ax2[0].axhline(y=0, ls=':', color='grey')
    ax2[0].set(ylabel='kg', xlabel=None)
    ax2[0].set_title('Free weight')
    sns.barplot(data=sm_metrics, y='value', x='model', hue='Metric', 
        errorbar=('se', 1.96),
        ax=ax2[1]).set_ylim([ymin, ymax])    
    ax2[1].axhline(y=0, ls=':', color='grey')

    # Label bars with value
    if annotate:
        for i in ax2[0].containers:
                ax2[0].bar_label(i, fmt='%.1f', label_type='center') 
        for i in ax2[1].containers:
                ax2[1].bar_label(i, fmt='%.1f', label_type='center') 
    # Titles and axis labels
    ax2[1].set(ylabel='kg', xlabel=None)
    ax2[0].set_title('Free weight')
    ax2[1].set_title('Smith machine')
    fig2.suptitle('Model evaluation metrics')
    return fig, fig2, fw_metrics, sm_metrics

def compare_models2(predictions, title='Measured 1RM vs. model predictions', 
    context='talk', annotate=True, ymin=-1.5, ymax=10):
    """2022-12-02 23:53 from `2022-12-02 iteration 4` notebook

    Plot predictions from all the models for a dataset.
        Parameters:
            - predictions (DataFrame): 
                Dataframe that contains target data ('Measured' column) and model predictions
                (1 column per model).
            - title (str): Overall plot title.
            - context (None or str): Seaborn .set_theme() parameter. 
                One of {paper, notebook, talk (default), poster}. If None, set to 'default (notebook)'.
            - annotate (bool): Whether or not to annotate the bar graph with values. Default is True.
        Returns:
            - figure with scatter plots of measured vs. predicted values for all models.
            - figure with bar charts of mean absolute error and mean error for all models.
            - DataFrame containing:
                mean absolute error and mean error for all models.

    Command syntax:
        scatterplot, error_plot, metrics = compare_models2(predictions, 
        title='Model predictions', context='talk')

        scatterplot.savefig('../output/figures/Measured vs predicted.png')
        error_plot.savefig('../output/figures/Error bar chart.png')
        path = r'../output/predictions'
        save_csv(metrics, 'Model errors', path=path)
    """
    fw_models = predictions[predictions.columns[~predictions.columns.str.contains('Measured')]].columns.to_list()
    # sns.reset_defaults()    
    # %matplotlib inline
    font_scale=.8 if context=='talk' else 1
    rc={'lines.markersize': 6} if context=='talk' else None
    sns.set_theme(context=context, style='ticks', font_scale=font_scale, 
        rc=rc)
    if (len(fw_models) == 6) | (len(fw_models) == 3):
        ncols=3
        nrows = round((len(fw_models)+1)//3)
    else:
        nrows = round((len(fw_models)+1)/4)
        ncols=4
    if nrows > 1:
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3.2))
    else:
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3.7))
    ax = ax.flatten()
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(1.4*len(fw_models), 4))
    fw_error = pd.DataFrame()
    for index, model in enumerate(fw_models):
        # Calculate error
        fw_error[model] = predictions[model] - predictions['Measured'] 

        # Plot measured vs. predicted values for each model
        ax[index].axline(xy1=(150, 150), slope=1, alpha=0.8, linewidth=0.5, color='orange')
        if index == 0:
            ax[index].set_ylabel('Predicted') 
        sns.scatterplot(data=predictions, x='Measured', y=model, ax=ax[index],
            alpha=0.5, marker='o', legend=False, 
        ).set(xlabel=None, ylabel=None)
        ax[index].set_title(model)
        if index % ncols == 0:
            ax[index].set_ylabel('Predicted 1RM') 
        if index >= len(fw_models) -ncols:
            ax[index].set_xlabel('Measured 1RM') 
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig2.tight_layout(rect=[0, 0, 1, 0.98])

    # Calculate remaining evaluation metrics and reshape dataframe for plotting
    fw_error['Metric'] = 'Error'
    fw_mae = abs(fw_error.iloc[:,:-1])
    fw_mae['Metric'] = 'MAE'
    fw_metrics = pd.concat([fw_error, fw_mae], axis=0).melt(
        value_vars=fw_models, id_vars=['Metric'], var_name='model')
    print(f'Metrics dataframe shape (free weight data): {fw_metrics.shape}')

    # Plot evaluation metrics: 
    sns.barplot(data=fw_metrics, y='value', x='model', hue='Metric', 
        errorbar=('se', 1.96), # error bars set to 95% confidence interval, or 1.96*standard error
        ax=ax2).set_ylim([ymin, ymax])
    ax2.axhline(y=0, ls=':', color='grey')
    ax2.set(ylabel='kg', xlabel=None)
    
    # Label bars with value
    if annotate:
        for i in ax2.containers:
                ax2.bar_label(i, fmt='%.1f', label_type='center') 
    # Titles and axis labels
    fig2.suptitle('Model evaluation metrics')
    return fig, fig2, fw_metrics