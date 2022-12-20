import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_cv_metrics(mae_df, r2_df, color='silver', context='talk', ymin=None, ymax=1.1):
    """
    Create barplots cross-validated MAE and r^2 for multiple models.

    Parameters:
        - mae_df, r2_df: DataFrames containing MAE and r^2, respectively,
            for all folds and all models (1 column per model).
    Returns:
        - Figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    # sns.reset_defaults()    
    # sns.set_theme(context=context, style='ticks')
    # %matplotlib inline
    fig, ax = plt.subplots(nrows=2, ncols=1, 
        figsize=(mae_df.shape[1]*(1.5 if context=='talk' else 1), 4))
    sns.barplot(data=mae_df, color=color, 
        ax=ax[0], errorbar=('se', 1.96)).set_title('Cross-Validated Mean Absolute Error')
    ax[0].set_ylabel('kg')
    for i in ax[0].containers:
        ax[0].bar_label(i, fmt='%.1f', label_type='center') 
    sns.barplot(data=r2_df, color=color, 
        ax=ax[1], errorbar=('se', 1.96)).set_title('Cross-Validated R^2 Score')
    ax[1].set_ylim([ymin, ymax])
    for i in ax[1].containers:
        ax[1].bar_label(i, fmt='%.3f', label_type='edge',padding=-25) 
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    # Titles and axis labels
    # fig.suptitle('Cross-Validated Evaluation Metrics')
    return fig

def plot_residuals2(predictions, title='Squat', 
    context='talk', annotate=True, ymin=-1.5, ymax=8, labels=None, pickle_name=None,
    path=r'C:\Users\silvh\OneDrive\lighthouse\projects\lighthouse-capstone-project\output\figures'):
    """
    Plot residuals from all the models for a dataset.
        Parameters:
            - predictions (DataFrame): 
                Dataframe that contains target data ('Measured' column) and model predictions
                (1 column per model).
            - title (str): Overall plot title.
            - context (None or str): Seaborn .set_theme() parameter. 
                One of {paper, notebook, talk (default), poster}. If None, set to 'default (notebook)'.
            - annotate (bool): Whether or not to annotate the bar graph with values. Default is True.
            - labels (list of strings): Model names for plot labels. 
                If None, labels will be from column names.
        Returns:
            - Figure of prediction residuals

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
        nrows = round((len(fw_models)+2)//3)
    else:
        nrows = round((len(fw_models)+3)//4)
        ncols=4
    fig3, ax3 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3))
    ax3 = ax3.flatten() if nrows > 1 else ax3
    fw_error = pd.DataFrame()
    fw_error['Measured'] = predictions['Measured'] 
    for index, model in enumerate(fw_models):
        # Calculate residual error
        fw_error[model] = predictions[model] - predictions['Measured'] 
        fw_error['Error direction'] = fw_error[model]/abs(fw_error[model]) # -1 or +1

        # Plot residuals
        ax3[index].axhline(y=0, alpha=0.9, linewidth=0.5, color='orange')
        sns.scatterplot(data=fw_error, y=model, 
            x='Measured', alpha=0.7, style='Error direction', markers={-1: 'v', 1: '^'},
            hue='Error direction', palette='coolwarm',
            ax=ax3[index], legend=False)
        ax3[index].set(title=(labels[index] if labels else fw_models[index]), 
            ylabel='Error (kg)' if (index % ncols == 0) else None, 
            xlabel='Measured 1RM' if index >= len(fw_models)-ncols else None)
    # Make y-axis range the same
    error_min = fw_error[fw_models].min().min()-5
    error_max = fw_error[fw_models].max().max()+5
    ax3 = [ax.set_ylim([error_min, error_max]) for ax in ax3]

    # Titles and axis labels
    fig3.suptitle(title+': Model Regression Residuals')
    fig3.tight_layout(rect=[0, 0, 1, 1])
    if pickle_name:    
        path = path + f'/{pickle_name}_'
        fig3.savefig(f'{path}residuals_plot_{title}.png')
        print('Figure saved: '+f'{path}residuals_plot_{title}.png')
    return fig3