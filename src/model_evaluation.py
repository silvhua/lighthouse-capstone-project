from sklearn.model_selection import cross_validate
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