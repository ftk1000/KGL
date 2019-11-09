# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:18:46 2019


C:\__KAGGLE__\Kaggle_Micro_Courses\intermediate_ml\acook_cat_vars_exc.py

https://www.kaggle.com/zurman/exercise-categorical-variables/edit


@author: Farid Khafizov
"""

'''
By encoding **categorical variables**, you'll obtain your best results thus far!
# Setup
The questions below will give you feedback on your work. Run the following cell 
to set up the feedback system.
'''

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex3 import *
print("Setup Complete")

'''

#%%
****In this exercise, you will work with data from the [Housing Prices 
Competition for Kaggle Learn Users]
(https://www.kaggle.com/c/home-data-for-ml-course). 

![Ames Housing dataset image](https://i.imgur.com/lTJVG4e.png)

Run the next code cell without changes to load the training and validation 
sets in `X_train`, `X_valid`, `y_train`, and `y_valid`.  
The test set is loaded in `X_test`.
'''
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('../input/train.csv', index_col='Id') 
X_test = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll drop columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()] 
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8, 
                                                      test_size=0.2,
                                                      random_state=0)

X_train.head()

#%%
'''
the dataset contains both numerical and categorical variables.  
You'll need to encode the categorical data before training a model.

To compare different models, you'll use the same `score_dataset()` function f
rom the tutorial.  This function reports the [mean absolute error](
https://en.wikipedia.org/wiki/Mean_absolute_error) (MAE) 
from a random forest model.
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

#%%   
# -------------------------- STEP 1 --------------------------
''' 
# Step 1: Drop columns with categorical data

You'll get started with the most straightforward approach.  Use the code cell 
below to preprocess the data in `X_train` and `X_valid` to remove columns with 
categorical data.  Set the preprocessed DataFrames to `drop_X_train` and 
`drop_X_valid`, respectively.  
'''
# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)

'''
Categorical variables:
['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 
'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 
'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition']

'''



# Fill in the lines below: drop columns in training and validation data
#drop_X_train = ____
#drop_X_valid = ____
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

# Check your answers
#step_1.check()

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

'''
MAE from Approach 1 (Drop categorical variables):
17837.82570776256
'''
#%%

# -------------------------- STEP 2 --------------------------
''' 
# Step 2: Label encoding
Before jumping into label encoding, we'll investigate the dataset.  
Specifically, we'll look at the `'Condition2'` column.  The code cell below 
prints the unique entries in both the training and validation sets.
'''
print("Unique values in 'Condition2' column in training data:", 
      sorted(X_train['Condition2'].unique()))
print("\nUnique values in 'Condition2' column in validation data:", 
      sorted(X_valid['Condition2'].unique()))
    
'''
Unique values in 'Condition2' column in training data: 
    ['Artery', 'Feedr', 'Norm', 'PosA', 'PosN', 'RRAe']

Unique values in 'Condition2' column in validation data: 
    ['Artery', 'Feedr', 'Norm', 'PosN', 'RRAn', 'RRNn']
'''

'''
This is a common problem that you'll encounter with real-world data, and there 
are many approaches to fixing this issue. For instance, you can write a custom 
label encoder to deal with new categories. The simplest approach, however, is 
to drop the problematic categorical columns.  

Run the code cell below to save the problematic columns to a Python list 
`bad_label_cols`.  Likewise, columns that can be safely label encoded are 
stored in `good_label_cols`.
'''

# All categorical columns
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if 
                   set(X_train[col]) == set(X_valid[col])]
        
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols)-set(good_label_cols))
        
print('Categorical columns that will be label encoded:', 
      sorted(good_label_cols))
'''
Categorical columns that will be label encoded: ['BldgType', 'CentralAir', 
'ExterQual', 'HouseStyle', 'KitchenQual', 'LandContour', 'LotConfig', 
'LotShape', 'MSZoning', 'PavedDrive', 'SaleCondition', 'Street']
'''


print('\nCategorical columns that will be dropped from the dataset:', 
      sorted(bad_label_cols))
'''
Categorical columns that will be dropped from the dataset: ['Condition1', 
'Condition2', 'ExterCond', 'Exterior1st', 'Exterior2nd', 'Foundation', 
'Functional', 'Heating', 'HeatingQC', 'LandSlope', 'Neighborhood', 'RoofMatl', 
'RoofStyle', 'SaleType', 'Utilities']
'''

'''
Use the next code cell to label encode the data in `X_train` and `X_valid`.  
Set the preprocessed DataFrames to `label_X_train` and `label_X_valid`, 
respectively.  
- We have provided code below to drop the categorical columns in 
   `bad_label_cols` from the dataset. 
- You should label encode the categorical columns in `good_label_cols`.  
'''

from sklearn.preprocessing import LabelEncoder

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

#Make copy to avoid changing original data 
#label_X_train = X_train.copy()
#label_X_valid = X_valid.copy()

# Apply label encoder 
#____ # Your code here

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in good_label_cols:
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])

# Check your answer
#step_2.b.check()

print("MAE from Approach 2 (Label Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
'''
MAE from Approach 2 (Label Encoding):
17575.291883561644
'''

#%%

# -------------------------- STEP 3 --------------------------
''' 
# Step 3: Investigating cardinality

So far, you've tried two different approaches to dealing with categorical 
variables.  And, you've seen that encoding categorical data yields better 
results than removing columns from the dataset.

Soon, you'll try one-hot encoding.  Before then, there's one additional topic 
we need to cover.  Begin by running the next code cell without changes.  
'''

# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])

'''
[('Street', 2),
 ('Utilities', 2),
 ('CentralAir', 2),
 ('LandSlope', 3),
 ('PavedDrive', 3),
 ('LotShape', 4),
 ('LandContour', 4), ...
'''

'''
The output above shows, for each column with categorical data, the number of 
unique values in the column.  For instance, the `'Street'` column in the 
training data has two unique values: `'Grvl'` and `'Pave'`, corresponding to a 
gravel road and a paved road, respectively.

We refer to the number of unique entries of a categorical variable as the 
**cardinality** of that categorical variable.  For instance, the `'Street'` 
variable has cardinality 2.

Use the output above to answer the questions below.
'''
# Fill in the line below: How many categorical variables in the training data
# have cardinality greater than 10?
high_cardinality_numcols = 3

# Fill in the line below: How many columns are needed to one-hot encode the 
# 'Neighborhood' variable in the training data?
num_cols_neighborhood = 25

# Check your answers
#step_3.a.check()

'''
For large datasets with many rows, one-hot encoding can greatly expand the size of the dataset.  For this reason, we typically will only one-hot encode columns with relatively low cardinality.  Then, high cardinality columns can either be dropped from the dataset, or we can use label encoding.

As an example, consider a dataset with 10,000 rows, and containing one categorical column with 100 unique entries.  
- If this column is replaced with the corresponding one-hot encoding, how many entries are added to the dataset?  
- If we instead replace the column with the label encoding, how many entries are added?  

Use your answers to fill in the lines below.
'''
# Fill in the line below: How many entries are added to the dataset by 
# replacing the column with a one-hot encoding?
OH_entries_added = 990000

# Fill in the line below: How many entries are added to the dataset by
# replacing the column with a label encoding?
label_entries_added = 0

# Check your answers
#step_3.b.check()


#%%

# -------------------------- STEP 4 --------------------------
''' 
# Step 4: One-hot encoding

In this step, you'll experiment with one-hot encoding.  
But, instead of encoding all of the categorical variables in the dataset, 
you'll only create a one-hot encoding for columns with cardinality less than 10.

Run the code cell below without changes to set 
`low_cardinality_cols` to a Python list containing the columns that will be 
one-hot encoded.  Likewise, `high_cardinality_cols` contains a list of 
categorical columns that will be dropped from the dataset.
'''

# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)

'''
Categorical columns that will be one-hot encoded: ['MSZoning', 'Street', 
'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1',
 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual',
 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual',
 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition']
Categorical columns that will be dropped from the dataset: ['Exterior2nd', 
 'Neighborhood', 'Exterior1st']
'''

from sklearn.preprocessing import OneHotEncoder

# Use as many lines of code as you need!
#OH_X_train = ____ # Your code here
#OH_X_valid = ____ # Your code here
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)



# Check your answer
#step_4.check()

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))

'''
MAE from Approach 3 (One-Hot Encoding):
17525.345719178084
'''



#%%

# -------------------------- STEP 5 --------------------------
'''
# Step 5: Generate test predictions and submit your results

After you complete Step 4, if you'd like to use what you've learned to submit your results to the leaderboard, you'll need to preprocess the test data before generating predictions.

**This step is completely optional, and you do not need to submit results to the leaderboard to successfully complete the exercise.**

Check out the previous exercise if you need help with remembering how to [join the competition](https://www.kaggle.com/c/home-data-for-ml-course) or save your results to CSV.  Once you have generated a file with your results, follow the instructions below:
- Begin by clicking on the blue **COMMIT** button in the top right corner.  This will generate a pop-up window.  
- After your code has finished running, click on the blue **Open Version** button in the top right of the pop-up window.  This brings you into view mode of the same page. You will need to scroll down to get back to these instructions.
- Click on the **Output** tab on the left of the screen.  Then, click on the **Submit to Competition** button to submit your results to the leaderboard.
- If you want to keep working to improve your performance, select the blue **Edit** button in the top right of the screen. Then you can change your model and repeat the process.
'''











