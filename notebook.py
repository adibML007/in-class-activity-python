#!/usr/bin/env python
# coding: utf-8

# ## In-Class Activity - Cyber Security Data Analysis 
# This notebook will guide you through the process of analyzing a cyber security dataset. Follow the TODO tasks to complete the assignment.
# 

# # Step 1: Importing the required libraries
# 
# TODO: Import the necessary libraries for data analysis and visualization.

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
pd.options.display.max_rows = 999
warnings.filterwarnings("ignore")


# # Step 2: Loading the dataset
# 
# TODO: Load the given dataset.

# In[17]:


link = "./Data/CySecData.csv"


# # Step 3: Display the first few rows of the dataset
# TODO: Import the necessary libraries for data analysis and visualization.

# In[18]:


df = pd.read_csv(link)
df.head()


# # Step 4: Initial info on the dataset.
# 
# TODO: Provide a summary of the dataset.

# In[19]:


# Display summary information about the dataset
df.info()
df.describe(include='all')
df['class'].value_counts()


# # Step 5: Creating dummy variables
# TODO: Create dummy variables for the categorical columns except for the label column "class".

# In[27]:


# Create dummy variables for categorical columns except for the label column "class"
categorical_cols = ['protocol_type', 'service', 'flag']
dfDummies = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
dfDummies.head()


# # Step 6: Dropping the target column
# TODO: Drop the target column 'class' from the dataset.

# In[ ]:


# Drop the target column 'class' from the dataset
dfDummies = dfDummies.drop('class', axis=1)
dfDummies.head()


# # Step 7: Importing the Standard Scaler
# TODO: Import the `StandardScaler` from `sklearn.preprocessing`.

# In[22]:


from sklearn.preprocessing import StandardScaler



# # Step 8: Scaling the dataset
# TODO: Scale the dataset using the `StandardScaler`.

# In[28]:


# Scale the dataset using StandardScaler
scaler = StandardScaler()
dfNormalized = scaler.fit_transform(dfDummies)
dfNormalized[:5]


# # Step 9: Splitting the dataset
# TODO: Split the dataset into features (X) and target (y).

# In[ ]:


# Split the dataset into features (X) and target (y)
X = dfDummies
y = df['class'].values


# # Step 10: Importing the required libraries for the models
# TODO: Import the necessary libraries for model training and evaluation.

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Now the required model libraries are imported and ready for use.


# # Step 11: Defining the models
# TODO: Define the models to be evaluated.

# In[ ]:


# Define the models to be evaluated
models = []
models.append(('LR', LogisticRegression(max_iter=1000)))
models.append(('SVM', SVC()))
models.append(('RandomForestClassifier', RandomForestClassifier()))

# # Step 12: Evaluating the models
# TODO: Evaluate the models using 10 fold cross-validation and display the mean and standard deviation of the accuracy.

# In[ ]:


# Evaluate the models using 10-fold cross-validation and display the mean and standard deviation of the accuracy
for name, model in models:
    scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")


# # Step 13: Converting the notebook to a script
# TODO: Convert the notebook to a script using the `nbconvert` command.

# In[ ]:


# Convert the notebook to a script using the nbconvert command
# get_ipython().system('jupyter nbconvert --to script "notebook.ipynb"')

