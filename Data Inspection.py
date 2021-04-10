import pandas as pd
import multiprocessing as mp
import sys

#######  PROJECT ###########
# Load the data 

import multiprocessing as mp 
import pandas as pd

vehicles = pd.read_csv("C:/Users/Carmen/OneDrive/Archivos - Todo/1 - Master Statistics/Period 2/Traffic_Crashes_-_Vehicles.csv")
crashes  = pd.read_csv("C:/Users/Carmen/OneDrive/Archivos - Todo/1 - Master Statistics/Period 2/Traffic_Crashes_-_Crashes.csv", sep=',')


# Just for confirming datasets' structure:
vehicles['VEHICLE_ID'].duplicated().any() # Primary Key 
vehicles['CRASH_RECORD_ID'].duplicated().any() # You have 
crashes['CRASH_RECORD_ID'].duplicated().any() # Primary Key 

# Proceeding to the data join (We're stil targeting behicles, but now have more columns)
df = pd.merge(vehicles, crashes, how = 'left', on = 'CRASH_RECORD_ID')

# No longer needed
del vehicles 
del crashes

# Inspection 
df.shape

# First Cleaning 
pd.set_option('display.max_rows', 140)
df.isnull().sum()


# Cleaning dataset




## I want to clear all the columns with no analytical utiliy or with a excessive ammount of empty values 

del df['CRASH_RECORD_ID']
del df['CRASH_UNIT_ID']
del df['VEHICLE_ID']

max_number_of_nas = 100000
set1 = df.loc[:, (df.isnull().sum() <= max_number_of_nas)]
set2 = df.loc[:, (df.isnull().sum() <= max_number_of_nas)]

pd.set_option('display.max_rows', 140)
df.isnull().sum()

# Descriptives 
from scipy.stats import pearsonr
pearsonr(data1, data2)

# ML 
import sklearn as sk
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit()

# decision tree: Explanation from the cscience professor




### Trail - Python - Test 
import pandas as pd

vehicles = pd.read_csv("C:/Users/Carmen/OneDrive/Archivos - Todo/1 - Master Statistics/Period 2/Traffic_Crashes_-_Vehicles.csv")
crashes  = pd.read_csv("C:/Users/Carmen/OneDrive/Archivos - Todo/1 - Master Statistics/Period 2/Traffic_Crashes_-_Crashes.csv", sep=',')
df = pd.merge(vehicles, crashes, how='left', on ='CRASH_RECORD_ID')

pd.set_option('display.max_rows', 140)
df.isnull().sum()

data = df.head(2000)  
View(data)

dat = data[['CRASH_HOUR', 'INJURIES_UNKNOWN', 'DAMAGE']]
dat.dropna(subset=['CRASH_HOUR', 'INJURIES_UNKNOWN', 'DAMAGE'], inplace=True)
dat.isnull().sum() # Awesome

x = dat[['CRASH_HOUR', 'INJURIES_UNKNOWN']]
y = dat[['DAMAGE']]

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

clf = RandomForestClassifier(random_state = 10, max_features='sqrt')
pipe = Pipeline([('classify', clf)])
param = {'classify__n_estimators':list(range(20, 30, 1)),
         'classify__max_depth':list(range(3, 10, 1))}
grid = GridSearchCV(estimator = pipe, param_grid = param, scoring = 'accuracy', cv = 10)
grid.fit(x, y)
print(grid.best_params_)
print(grid.best_score_)


### Titanic data

train_df = pd.read_csv("C:/Users/Carmen/OneDrive/Archivos - Todo/1 - Master Statistics/Period 2/train.csv")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

train_df["Age"].fillna(train_df.Age.mean(), inplace=True)
train_df["Embarked"].fillna("S", inplace=True)
train_df.isnull().sum()

x_train = train_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Parch', 'SibSp']]
x_train = pd.get_dummies(x_train)

y_train = train_df[['Survived']]

clf = RandomForestClassifier(random_state = 10, max_features='sqrt')
pipe = Pipeline([('classify', clf)])
param = {'classify__n_estimators':list(range(20, 30, 1)),
         'classify__max_depth':list(range(3, 10, 1))}
grid = GridSearchCV(estimator = pipe, param_grid = param, scoring = 'accuracy', cv = 10)
grid.fit(x_train, y_train)
print(grid.best_params_)
print(grid.best_score_)

# End 





from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit()





type(x)
type(y['DAMAGE'])





# 


## Trainning 

- Tunning 

"Skicit learn works with numpy by defual"

"In numpy, categorical variables need to be defines as integers or dummies(one.hot-encoding)"

"NAs are np.na"

"If you use NumPy you loss the names because you use nnumpy matrices"

""

## Data Partition

x_train = df[['LONGITUDE','MAKE']]
y_train = df['DAMAGE']

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')

clf = RandomForestClassifier(random_state = 10, max_features='sqrt', n_jobs = 5)
pipe = Pipeline([('classify', clf)])
param = {'classify__n_estimators':list(range(20, 30, 1)),
         'classify__max_depth':list(range(3, 10, 1))}
grid = GridSearchCV(estimator = pipe, param_grid = param, scoring = 'accuracy', cv = 10)

grid.fit(x, y)
print(grid.best_params_)
print(grid.best_score_)


## Model Evaluation 

# The idea is to identify the model with the highest accuracy. I a variable is an execellent predictor,
# Even though, it has 80% empty values, we migth considered keeing it in!









# non useful variables: 


ID 












