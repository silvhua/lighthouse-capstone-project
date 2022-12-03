# Goals

Maximal strength is measured using the maximum weight that one can use for a given exercise, known as the **1 repetition maximum (1RM)**. 1RM strength is the key performance indicator by which Olympic weightlifters and powerlifters are evaluated. Other athletes and exercisers may also want to estimate 1RM strength in order to:
* Determine how much weight to use on a given training session. It is common for structured training programs to prescribe weight selection based on a percentage of 1RM.
* Enhance motivation, particularly with number-oriented individuals.

Testing one's 1RM, which involves progressively increasing weight on a given exercise until reaching the point of failure, is inherently fatiguing and higher risk, particularly for an exercise such as the barbell squat. Additionally, achieving one's true 1RM on a given day is contingent on having optimal conditions such as low fatigue and the right psychological conditions. Whereas strength athletes test 1RM only a few times per year, often with a period of lower training volume in the preceding weeks to allow fatigue to dissipate for full expression of strength, submaximal testing can be performed frequently. 

For the reasons outlined above, competitive athletes and exercisers benefit from estimating their maximal strength (as indicated by 1RM) using submaximal loads (e.g. 80%). Athletes and exercisers use weight lifted to evaluate strength as it is an objective indicator of how much force the body exerts. **Concentric velocity** (i.e. movement speed when lifting the weight) is an additional objective indicator of force: Being able to move the same weight at a faster speed means that more force is being applied. Athletes who track their concentric velocities during testing with 1RM and submaximal loads can then create a **load-velocity profile** (see figure 1 below) using statistical linear regression to then estimate 1RM in the future using:
* Slope of the load-volocity profile (**LV slope**).
* Intercept of the load-velocity profile (**LV intercept**).
* Estimated or previously-measured velocity at 100% of the 1RM, known as the **minimum velocity threshold (MVT)**

Subsequent prediction of 1RM using a load-velocity profile assumes that even when the athlete's strength changes (which changes the LV slope and intercept) the MVT is constant. 

The goal of this project is to use machine learning (ML) to develop a regression model for predicting 1 repetition maximum (1RM) strength for the free weight squat and Smith machine squat using submaximal weight, even without knowledge of MVT (unlike with statistical regression methods). This was done using load and velocity data during repetitions performed at submaximal loads with the intent to move at maximum concentric velocity. 

# Hypothesis

This project is based on a previous study by [Balsalobre-Fernández and Kipp (2021)](https://doi.org/10.3390/sports9030039), which had the same objective using the bench press exercise. Based on their findings, I hypothesize that:
* Machine learning models can allow for 1RM estimation for both the free weight (FW) squat and Smith machine (SM) squat using two features, the slope and y-intercept of the individual's load-velocity profile, without knowledge of the concentric velocity during a 1RM load (MVT). 
* Linear regression machine learning (ML) models will allow for 1RM estimation with a lesser likelihood of overestimation than statistical machine learning models

# Exploratory Data Analysis
## The Data Set
Data for this project were provided by Dr. Carlos Balsalobre-Fernández from Universidad Autónoma de Madrid. Data were collected by *[name of authors who performed the experiments]* with 52 participants who each performed the FW squat and SM squat at various loads. Participants were instructed to perform 1-2 repetitions with their maximal intended velocity. There were 2 data sets (1 for each of the FW squat and SM squat), each with 52 rows (1 per participant). The columns are as follows:
* `Participant ID`
* `Age`
* `Mass`
* `Height`
* `Load20%1RM` (weight in kg)
* `Load40%1RM`
* `Load60%1RM`
* `Load80%1RM`
* `Load90%1RM`
* `Load-1RM-1` (target variable)
* `20% MV` (`MV` = mean concentric velocity, in m/s)
* `40%MV`
* `60%MV`
* `80%MV`
* `90%MV`
* `100%MV`

## Data Visualization
### Original Features
Load-velocity profiles for all subjects are shown in this figure:
![visualization of all LV profiles](./output/figures/data_viz.png)

Univariate distributions and bivariate correlations of the features used for modelling are shown in the pairplots for the FW and SM squats below:















### Engineered Features
#### FW squat

<img src="./output/figures/04 iteration FW pairplot.png" width=400>

#### SM squat

<img src="./output/figures/04 iteration SM pairplot.png" width=400>

Limitations of the data:
* The variables are not normally distributed. It is likely that with a larger sample size, the data would have a normal distribution. I decided not to transform the data to to reduce skewness because it would make the results of the models less practical to apply for athletes and coaches. 




# Iteration 1
This project was performed using Python in VSCode with the Jupyter Notebook extension. Modules and packages include Pandas, Numpy, Matplotlib, Seaborn, Scikit-Learn, and Keras/Tensorflow. 


## Data Preparation

Minimal data cleaning were required as most of this was done by the researchers who conducted the experiments. During the exploratory data analysis, a data entry error was identified and corrected.

<details>
<summary>Expand for details of the data entry error. </summary>
The `Load80%1RM` feature used during data collection should be ~80% of the participant's `Load-1RM-1`. For one participant, the value was much lower than expected (0.8 x 197.5 = 177.75, but the value listed was 117.5), likely due to a data entry error. This incorrect value was replaced with 177.5 (closest weight increment to 80% of the `Load-1RM-1`) 
</details>



## Feature Engineering

For each individual and exercise, the slope  (`LV slope`) and y-intercept (`LV intercept`) of the load-velocity profile were obtained using a linear regression (Scikit-Learn's `LinearRegression` class) using loads equal to 40%, 60%, and 80% of the individual's 1RM (with load as the dependent variable of the regression). 

Below are the load velocity profiles and linear regression predictions for 10 participants:
![individual LV profiles](./output/figures/EDA%20figures/04%20iteration%20sample%20LV%20profiles%20FW.png)



























## Modelling

The project aimed to find the model that would predict 1RM with the least error and with least likelihood of overestimation. Simple models were preferred to facilitate practical application by athletes and coaches.  Thus, linear regression models were selected over other regression models. 
The first iteration of the project aimed to replicate the 5 models used in the paper by Balsalobre-Fernández and Kipp (2021):

Model | Description | Model Independent Variable(s) 
--- | ---- | ---
Model 1 | Statistical linear regression | **individual** MVT
Model 2 | Statistical linear regression | **mean group** MVT
Model 3 | ML multilinear regression with ordinary least squares (OLS) | LV slope and LV intercept
Model 4 | ML Lasso regression | LV slope and LV intercept
Model 5 | Neural network with 1 hidden layer of 10 nodes | LV slope and LV intercept

The second iteration of the project used the ML linear regression with OLS (Model 3). 
<br>

### Statistical Models (Models 1 & 2)
<br>

1RM was predicted using the following equation:
$$ 
1RM = LV_{slope} \times MVT + LV_{intercept}
$$ 

For model 1, MVT was the mean velocity measured for the individual using the 1RM load. For model 2, MVT was the mean velocity for all participants for each exercise (0.27519 m/s for FW squat, 0.25558 m/s for SM squat)



<br>

### Machine Learning Models (Models 3-4)
<br>

ML learning regressions were implemented using Scikit-Learn's `LinearRegression` (Model 3, OLS) and `LassoCV` (Model 4) classes. Five-fold cross validation (CV) with 100 `alpha` values was used to determine the Lasso regression model regularization strength to use for model fitting. The alpha values of the optimized models are shown below:

Exercise | Alpha | CV iterations
--- | ---- | ---
FW squat | 1.11 | 72
SM squat | 1.11 | 72

With the exception of model 5, the results for Iteration 1 are from models fit on data for all participants. In subsequent iterations, models were evaluated using 10-fold cross validation.

<br>

### Neural Network
<br>

A Keras/Tensorflow neural network was compiled using a `Sequential` model using following architecture:
1. `Normalization` layer that receives 2 inputs, the LV slope and LV intercept, and normalizes the data.
2. `Dense` layer with `sigmoid` activation.
3. `Dense` hidden layer with 10 neurons with linear output.
4. `Dense` output layer with 1 neuron.

The code for the network is shown below:
```python
model = Sequential()
if normalize:
    model.add(Normalization())
model.add(Dense(10, activation='sigmoid', input_shape=(X_test.shape[1],))) # Outputs to 10 hidden neurons
model.add(Dense(1))
model.compile(
    loss='mean_absolute_error',
    optimizer='adam',
    metrics=['mean_absolute_error']
)
```
The model was trained using the train sample and validated using the test sample (42 train samples, 10 test samples). To ensure that the train and test samples had similar mean target values, participants were ordered sequentially based on 1RM load on the free weight squat. Then, every 5th participant from the sample was selected for the test group. 

## Results
### Models 1-4
The 1RM predictions are plotted against the measured 1RM values in this along with the equality line:
![predicted vs measured values](./output/figures/04%20Iteration%20Measured%20vs%20predicted%20for%20all%20samples.png)

A few observations can be seen at first glance:
* Predictions from all models correlate highly with measured values. 
* There was more error with the SM squat predictions than with the FW squat predictions


Because the Pearson correlation coefficients (r) and coefficient of determination ($ R^2 $) for all models were similarly high (0.98-1.0), more practical evaluation metrics would be:
* Mean error: $ y_{predicted} - y_{measured} $
* Mean absolute error (MAE): $ | y_{predicted} - y_{measured} | $

The graphs below display these evaluation metrics for each model and exercise:
![evalutation metrics](./output/figures/04%20Iteration%20Error%20bar%20chart%20for%20all%20samples.png)











Based on the error values (blue bars), the statistical models (`Stat ind MVT` and `Stat grp MVT`) on average overestimate 1RM by 2.6 kg for each of the FW and SM squat, whereas the ML models (`OLS` and `Lasso`) are as equally likely to overpredict as they are to underpredict. 

Based on mean absolute error (MAE; red bars), ML models performed slightly better than the statistical models (error of 2.0-3.4 kg vs. 3.0-4.4 kg).

### Neural Network (Model 5)

The neural network was trained using the Adam algorithm (a stochastic gradient descent method) to minimize mean absolute error and to stop once this loss metric stopped improving after 50 epochs. Below are the results of this model:

<img src="./output/figures/individual%20model%20figures/sm_model5_history_test.png" width=300>
<img src="./output/figures/individual%20model%20figures/sm_model5_test.png" width=300>

Given that all predictions were almost the same value, this model's predictions are not of practical use.

# Iteration 2
Given that the ML models used in this project only required the LV slope and LV intercept, this can theoretically be calculated using two data points per participant. This iteration focused on FW squats, which are more commonly performed than SM squats. This time, instead of determining LV slope and LV intercept based on the entire range of %1RM available, these were determined on a subset of the loads (e.g. 40% and 60% 1RM).

The aims of this iteration are
1. To determine which two loads would provide the LV slope and LV intercepts leading to the 1RM predictions with the least error. 
2. Whether prediction error is affected by the number of data points used for estimating an individual's LV slope and LV intercept

It is hypothesized that:
1. Using moderate to heavy loads will provide the best estimates, since there will be less variability in the concentric velocity (as indicated in Figure 1, where there is greater variability in the data at lighter loads).
2. If the appropriate two loads are selected, using more data points to calculate LV slope and LV intercept won't meaningfully improve 1RM prediction.

## Process
Each individual's LV slope and LV intercept were calculated using each of the following subsets of %1RM loads:
* 20% and 60%
* 20% and 80%
* 20% and 90%
* 40% and 60%
* 40% and 80%
* 60% and 80%
* 60% and 90%
* 80% and 90%
* 40%, 60, and 80%
* 40%, 60, 80, and 90%

Each set of LV slopes and LV intercepts were then used as features to fit OLS linear regression models.

## Results
As hypothesized, predictions had lowest error when at least one of the loads used to calculate LV slope and LV intercept was 80% of 1RM or higher: Error was 6.4-6.8 kg when all loads were < 80% 1RM vs. 2.9-3.5 kg when at least one of the loads was 80% or higher. The magnitude of the other load (e.g. 20% vs. 60%) had no meaningful impact. Additonally, as long as LV slope and LV intercept were calculated with one of the weights being ~80+ or more, using more than two data points did not meaningfully improve model predictions (error for the `LV 40-60-80-90` model was 2.8 kg vs. 2.9 kg for `LV 40-80`).


<img src="./output/figures/02%20iteration%20measured%20vs%20predicted%20for%20all%20samples%20SELECT%20MODELS.png" width=600>

<img src="./output/figures/02%20iteration%20error%20bar%20chart%20for%20all%20samples%20SELECT%20MODELS.png" width=600>

Select models were evaluated with the coefficient of determination (r^2) and mean absolute error (MAE) using 10-fold cross-validation. As expected, MAE values determined through cross-validation were higher, but not to a meaningful amount. Results are as follows:






















. | LV 20-60 | LV 20-80 | LV 20-90 | LV 40-60 | LV 40-80 | LV 60-80 | LV 60-90 | LV 80-90 | LV 40-60-80 | LV 40-60-80-90
---| ---| ---|---|---|---|---|---|---|---|---|
mae | 7.433 | 3.347 | 3.652 | 7.040 | 3.037 | 3.177 | 3.403 | 3.068 | 3.446 | 3.011
r^2 | 0.825 | 0.969 | 0.941 | 0.846 | 0.973 | 0.973 | 0.957 | 0.965 | 0.970 | 0.970





















# Conclusions
Machine learning linear regressions allow athletes to predict 1RM using submaximal testing simply based on the LV profile; no estimation of minimum velocity threshold or prior 1RM testing is required. These ML models perform at least as equally well as the statistical regression models that require MVT, and are less likely to overestimate 1RM for a given participant. The OLS and Lasso linear regressions performed equally well to each other. Furthermore, LV profile can simply be estimated using two data points as long as one of the loads is at least ~80%.  The follow regression equations can be used to estimate 1RM using these models *[coefficients `a` and `b` to be filled in later]*:

Model | FW Squat | SM Squat
--- | ---- | ---
OLS | a × LV_{slope} + b × LV_{intercept} | a × LV_{slope} + b × LV_{intercept} 
Lasso | a × LV_{slope} + b × LV_{intercept} | a × LV_{slope} + b × LV_{intercept}

Given errors of these models, an athlete who wants to be conservative can subtract 2-3 kg (or more) from the estimated 1RM value.

Using these trained models, web app has been created for users who want to use their load-velocity data to predict 1RM, or simply to compute and visualize the slope and intercept of their load-velocity curve. Users can input their data for 2-4 loads, and one of the train models will provide a prediction based on LV slope and LV intercept determine from the linear regression from the entered data. The URL for the web app is https://silvhua-lighthouse-capstone-project-srcapp-wafhso.streamlit.app/ 
*[Provide details of the deployed models]*

# Challenges
The nature of most exercise science research is that it is challenging to recruit a large number of participants. After discussing with a program mentor about the poor performance of the neural network model, I decided to focus on linear regression models, as neural networks require thousands of samples to work well. At the same time, having prediction models that don't require neural networks makes their use more accessible for the population at large.

Another limitation of the data is that the results will likely be different for individuals who are not similar to the population in this study: males ages 18-26 (21.7 ± 2.0 years) with a mean 1RM FW squat of 136.3 ± 27.0 kg.

# Future Goals

Model accuracy would likely improve if accounting for other features of the participants, such as sex, experience level with the exercise, and anthropometrics. Additional research would be required to develop models for using LV profiles to predict 1RM strength for other exercises.