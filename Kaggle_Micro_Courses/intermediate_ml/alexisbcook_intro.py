# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:01:55 2019
C:\__KAGGLE__\Kaggle_Micro_Courses\intermediate_ml\alexisbcook_intro.py

ref:
https://www.kaggle.com/alexisbcook/introduction

@author: Farid Khafizov
"""

'''
Welcome to Kaggle Learn's Intermediate Machine Learning micro-course!
If you have some background in machine learning and you'd like to learn how to 
quickly improve the quality of your models, you're in the right place! 
In this micro-course, you will accelerate your machine learning expertise 
by learning how to:
 - tackle data types often found in real-world datasets (missing values, categorical variables),
 - design pipelines to improve the quality of your machine learning code,
 - use advanced techniques for model validation (cross-validation),
 - build state-of-the-art models that are widely used to win Kaggle competitions (XGBoost), and
 - avoid common and important data science mistakes (leakage).
'''

#%%
'''
Your Turn
Continue to the first exercise to learn how to submit predictions to a Kaggle 
competition and determine what you might need to review before getting started.
'''
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex1 import *
print("Setup Complete")
#%%
'''
You will work with data from the [Housing Prices Competition for Kaggle Learn 
Users](https://www.kaggle.com/c/home-data-for-ml-course) to predict home 
prices in Iowa using 79 explanatory variables describing (almost) every 
aspect of the homes.  

![Ames Housing dataset image](https://i.imgur.com/lTJVG4e.png)

Run the next code cell without changes to load the training and validation 
features in `X_train` and `X_valid`, along with the prediction targets in 
`y_train` and `y_valid`.  The test features are loaded in `X_test`.  
(_If you need to review **features** and **prediction targets**, please 
check out [this short tutorial](
https://www.kaggle.com/dansbecker/your-first-machine-learning-model).  
To read about model **validation**, look [here]
(https://www.kaggle.com/dansbecker/model-validation).  
Alternatively, if you'd prefer to look through a full course to review all 
of these topics, start [here](https://www.kaggle.com/learn/machine-learning).)_
'''


import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                      y, 
                                                      train_size=0.8, 
                                                      test_size=0.2,
                                                      random_state=0)

X_train.head()

#%%
'''
Step 1: Evaluate several models

The next code cell defines five different random forest models.  
Run this code cell without changes.  (_To review **random forests**, 
look [here](https://www.kaggle.com/dansbecker/random-forests)._)
'''
from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

'''
To select the best model out of the five, we define a function `score_model()` 
below.  This function returns the mean absolute error (MAE) from the 
validation set.  Recall that the best model will obtain the lowest MAE.  
(_To review **mean absolute error**, look [here]
(https://www.kaggle.com/dansbecker/model-validation).)_
'''

from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
    
# Fill in the best model
best_model = model_3

# Check your answer
#step_1.check()

#%%
'''
# Step 2: Generate test predictions
You know how to evaluate what makes an accurate model. 
Now it's time to go through the modeling process and make predictions. In the 
line below, create a Random Forest model with the variable name `my_model`.
'''

# Define a model
my_model = model_3 # Your code here

# Fit the model to the training data
my_model.fit(X, y)

# Generate test predictions
preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

#%%
'''
# Step 3: Submit your results

Once you have successfully completed Step 2, you're ready to submit your 
results to the leaderboard!  First, you'll need to join the competition if you 
haven't already.  So open a new window by clicking on 
[this link](https://www.kaggle.com/c/home-data-for-ml-course).  
Then click on the **Join Competition** button.

![join competition image](https://i.imgur.com/wLmFtH3.png)

Next, follow the instructions below:
- Begin by clicking on the blue **COMMIT** button in the top right corner of 
    this window.  This will generate a pop-up window.  
- After your code has finished running, click on the blue **Open Version** 
    button in the top right of the pop-up window.  This brings you into view 
    mode of the same page. You will need to scroll down to get back to these 
    instructions.
- Click on the **Output** tab on the left of the screen.  Then, click on the 
   **Submit to Competition** button to submit your results to the leaderboard.
- If you want to keep working to improve your performance, select the blue 
   **Edit** button in the top right of the screen. Then you can change your 
   model and repeat the process.
'''


