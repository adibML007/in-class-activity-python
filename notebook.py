#!/usr/bin/env python
# coding: utf-8

# ## In-Class Activity - Cyber Security Data Analysis 
# This notebook will guide you through the process of analyzing a cyber security dataset. Follow the TODO tasks to complete the assignment.
# 

# # Step 1: Importing the required libraries
# 
# TODO: Import the necessary libraries for data analysis and visualization.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
pd.options.display.max_rows = 999
warnings.filterwarnings("ignore")


# # Step 2: Loading the dataset
# 
# TODO: TODO: Load the given dataset.

# In[ ]:


link = "./Data/CySecData.csv"
df = pd.read_csv(link)


# # Step 3: Display the first few rows of the dataset
# 
# TODO: Display the first few rows of the dataset to understand its structure.

# In[ ]:


df.head()


# # Step 1: Importing the required libraries
# 
# TODO: Import the necessary libraries for data analysis and visualization.

# In[ ]:


df.info()


# # Step 1: Importing the required libraries
# 
# TODO: Import the necessary libraries for data analysis and visualization.

# In[ ]:


dfDummies = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])


# # Step 1: Importing the required libraries
# 
# TODO: Import the necessary libraries for data analysis and visualization.

# In[ ]:


dfDummies = dfDummies.drop(['class'], axis=1)


# # Step 7: Importing the Standard Scaler
# TODO: Import the `StandardScaler` from `sklearn.preprocessing`.

# In[ ]:


from sklearn.preprocessing import StandardScaler


# # Step 8: Scaling the dataset
# TODO: Scale the dataset using the `StandardScaler`.

# In[ ]:


scaler = StandardScaler()
scaler_df = scaler.fit_transform(dfDummies)
dfNormalized = pd.DataFrame(scaler_df, columns=dfDummies.columns)
dfNormalized


# # Step 9: Splitting the dataset
# TODO: Split the dataset into features (X) and target (y).

# In[ ]:


X = dfNormalized
y = df['class']


# # Step 10: Importing the required libraries for the models
# TODO: Import the necessary libraries for model training and evaluation.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# # Step 11: Defining the models
# TODO: Define the models to be evaluated.

# In[ ]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('SVM', SVC()))
models.append(('RandomForestClassifier', RandomForestClassifier()))


# # Step 12: Evaluating the models
# TODO: Evaluate the models using cross-validation and display the mean and standard deviation of the accuracy.

# In[ ]:


from numpy import mean
from numpy import std
for name, model in models:
    kfold = KFold(n_splits=2, random_state=5, shuffle=True)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(f'Model:{name}, accuracy:{cv_results}, mean: {mean(cv_results)}, std: {std(cv_results)}')


# # Step 13: Converting the notebook to a script
# TODO: Convert the notebook to a script using the `nbconvert` command.

# In[ ]:


get_ipython().system('jupyter nbconvert --to python notebook.ipynb')

