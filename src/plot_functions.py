
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LinearRegression
import numpy as np
import re
from processing_functions import *

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
    
    sns.reset_defaults()    
    %matplotlib inline
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
    sns.reset_defaults()    
    %matplotlib inline
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