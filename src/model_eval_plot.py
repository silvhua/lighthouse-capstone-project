import matplotlib.pyplot as plt
import seaborn as sns

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