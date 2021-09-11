#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


data = pd.read_csv("heart.csv")
data.head(10)


# In[3]:


#1 : Male, 0 : Female
sns.countplot(x="sex", data=data)


# In[4]:


#cp : Chest Pain
sns.countplot(x="cp", data=data)


# In[5]:


#1 : Yes, 0 : No
sns.countplot(x="exang", data=data)


# In[6]:


#Exercise Induced Angina
sns.countplot(x="exang", data=data)


# In[7]:


#fbs : Fasting Blood Pressure
sns.countplot(x="fbs", data=data)


# In[8]:


#restecg : Resting Electrocardiographic Results
sns.countplot(x="restecg", data=data)


# In[9]:


#fbs : Fasting Blood Pressure, exang : Exercise Induced Angina
sns.barplot(x="fbs", y="exang", data=data)


# In[10]:


# Finding the Null values
data.isnull()


# In[11]:


data.isnull().sum()


# In[23]:


sns.heatmap(data.isnull(), yticklabels=False, cmap="viridis")


# In[12]:


#Statistical measures about the data
data.describe()


# In[13]:


#Checking the distribution of Target Variables
data['target'].value_counts()
#1 = Defective Heart
#0 = Healthy Heart


# In[14]:


#Splitting the features and Target
x = data.drop(columns = 'target', axis = 1)
y = data['target']
print(x)


# In[15]:


print(y)


# In[21]:


#Splitting the Data into Training Data and Test Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = )
print(x.shape, x_train.shape, x_test.shape)


# In[17]:


#Model Training : Logistic Regression
model = LogisticRegression()
#training the LogisticRegression model with Training Data
model.fit(x_train, y_train)


# In[18]:


#Model Evaluation : Accuracy Score
#Accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy on Training Data: ', training_data_accuracy)


# In[19]:


#Accuracy on Test Data
x_test_prediction = model.predict(x_test)
test_data_accuracy =  accuracy_score(x_test_prediction, y_test)
print('Accuracy on Test Data: ', test_data_accuracy)


# In[20]:


#Building a Predictive System
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)

#change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 0):
    print('The Person does not have a heart disease')
else:
    print('The person has heart disease')

