import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# App link: https://silvhua-lighthouse-capstone-project-srcapp-wafhso.streamlit.app/
st.title('1RM Estimator')
st.markdown('### Use your load-velocity profile to estimate your back squat 1RM')
st.markdown('')
st.markdown('#### Please enter data for at least two different loads (see left sidebar).')
st.markdown('')
st.markdown('Created by [Silvia Hua](https://www.linkedin.com/in/silviahua/)')
st.markdown('Project details found on [Github](https://github.com/silvhua/lighthouse-capstone-project)')

def app_linear_regression(loads, velocities):
    """2022-11-30 19:00
    Calculate slope and intercept using linear regression, where X = load, y = velocity.
    """
    from sklearn.linear_model import LinearRegression
    
    loads = np.asarray(loads).reshape(-1,1)
    velocities = np.asarray(velocities).reshape(-1,1)
    lr = LinearRegression()
    lr.fit(velocities, loads)
    
    X = pd.DataFrame()
    X.loc[0,'slope'] = lr.coef_[0][0] 
    X.loc[0,'intercept'] = lr.intercept_[0]

    return X

def plot_lv_profile(loads, velocities, X):
    fig = go.Figure()
    if len(loads) > 2:
        y_pred = np.asarray(velocities) * X.loc[0, 'slope'] + X.loc[0, 'intercept']
        fig.add_trace(go.Scatter(x=velocities, 
        y=y_pred,
        name='Best fit line', mode='lines',
        line=dict(dash='dot', color='grey'), opacity=0.5,
        ))
    # Sort that data so that it will be plotted in sequence
    data = pd.DataFrame([loads, velocities], 
        index=['loads', 'velocities']).transpose().sort_values('velocities')

    fig.add_trace(go.Scatter(x=data['velocities'], y=data['loads'],
        mode='lines+markers', name='Entered data',
        ))
    fig.update_layout(
        title='Load velocity profile', xaxis_title='Velocity', yaxis_title='Load',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

velocities = []
loads = []
weight1 = st.sidebar.number_input('Weight #1')
velocity1 = st.sidebar.number_input('Mean concentric velocity #1')
weight2 = st.sidebar.number_input('Weight #2')
velocity2 = st.sidebar.number_input('Mean concentric velocity #2')
weight3 = st.sidebar.number_input('Weight #3 (optional)')
velocity3 = st.sidebar.number_input('Mean concentric velocity #3 (optional)')
weight4 = st.sidebar.number_input('Weight #4 (optional)')
velocity4 = st.sidebar.number_input('Mean concentric velocity #4 (optional)')
if weight1 > 0:
    loads.append(weight1)
    if weight2 > 0:
        loads.append(weight2)
        if weight3 > 0:
            loads.append(weight3)
            if weight4 > 0:
                loads.append(weight4)
if velocity1 > 0:
    velocities.append(velocity1)
    if velocity2 > 0:
        velocities.append(velocity2)
        if velocity3 > 0:
            velocities.append(velocity3)
            if velocity4 > 0:
                velocities.append(velocity4)

# Only take the number of data points where both velocity and load are provided
n_data_points = min(len(loads), len(velocities))
loads = loads[:n_data_points]
velocities = velocities[:n_data_points]

st.write(f'Loads entered: {loads}')
st.write(f'Velocities entered: {velocities}')

if (len(loads) > 1) & (len(velocities) == len(loads)):
    X = app_linear_regression(loads, velocities)
    if len(loads) == 2:
        model = pickle.load(open('output/models/02 iteration model40_80.sav', 'rb'))
    elif len(loads)==3:
        model = pickle.load(open('output/models/02 iteration model40_60_80.sav', 'rb'))
    else:
        model = pickle.load(open('output/models/02 iteration model40_60_80_90.sav', 'rb'))
    """
    ## Results
    """
    st.subheader(f"**Estimated 1RM: {round(model.predict(X)[0],1)}**")

    plot_lv_profile(loads, velocities, X)
    st.write(f"Load-velocity slope: \t\t{round(X['slope'].values[0],1)}")
    st.write(f"Load-velocity y-intercept: \t\t{round(X['intercept'].values[0],1)}")