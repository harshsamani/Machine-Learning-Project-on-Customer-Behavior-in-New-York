#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


taxi = pd.read_csv('/Users/harshsamani/Desktop/Masters/Data mangement & Big data/green_tripdata_2018-06.csv')


# In[3]:


taxi.head(5)


# Data Preprocessing and Understanding the Data

# In[4]:


taxi.describe()


# Finding the total number of Null values in the dataset

# In[5]:


taxi.isnull().sum()


# Number of records stored by each LPEP provider <br>
# where 1 = Creative Mobile Technologies & 2= VeriFone Inc.

# In[6]:


taxi['VendorID'].value_counts()


# In[7]:


# Dropping the entire column ehail_fee which contained the highest number of null values (entire column).
taxi.drop(['ehail_fee'], axis=1,inplace = True)


# In[8]:


#checking the dataset by looking at first 5 values.
taxi.head()


# Dropping the columns which are not require to answer business questions

# In[9]:


taxi.drop(['store_and_fwd_flag','extra','mta_tax','tolls_amount','improvement_surcharge'], axis=1, inplace=True)
taxi.head(5)


# Making the datetime column to datetime format by changing it's format from object

# In[10]:


# The column containing datetime as object datatype which needs to be changed in datetime datatype for better results.
taxi['lpep_pickup_datetime'].dtype


# In[11]:


# Making the columns into datatime datatype
taxi['lpep_pickup_datetime'] = pd.to_datetime(taxi['lpep_pickup_datetime'])
taxi['lpep_pickup_datetime'].dtype


# In[12]:


# Making the datatype object to datatime for the column lpep_dropoff_datetime
taxi['lpep_dropoff_datetime'] = pd.to_datetime(taxi['lpep_dropoff_datetime'])
taxi['lpep_dropoff_datetime'].dtype


# In[13]:


# splitting the lpep_pickup_datetime to a seperate column into hours, month aand day of the week.
taxi['Hour_pickup'] = taxi['lpep_pickup_datetime'].apply(lambda time: time.hour)
taxi['Month_pickup'] = taxi['lpep_pickup_datetime'].apply(lambda time: time.month)
taxi['Day of Week_pickup'] = taxi['lpep_pickup_datetime'].apply(lambda time: time.dayofweek)


# In[14]:


# Using the .map() function with this dictionary to map the actual string names to the day of the week:
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
taxi['Day of Week_pickup'] = taxi['Day of Week_pickup'].map(dmap)
taxi.head(5)


# In[15]:


#Now performing same task for lpep_dropoff_datetime column and making into seperate column hours, month aand day of the week.
taxi['Hour_drop'] = taxi['lpep_dropoff_datetime'].apply(lambda time: time.hour)
taxi['Month_drop'] = taxi['lpep_dropoff_datetime'].apply(lambda time: time.month)
taxi['Day of Week_drop'] = taxi['lpep_dropoff_datetime'].apply(lambda time: time.dayofweek)


# In[16]:


# Again use of the .map() function with this dictionary to map the actual string names to the day of the week:
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
taxi['Day of Week_drop'] = taxi['Day of Week_drop'].map(dmap)
taxi.head(5)


# <h2> Analyzing Trends
# <h4> For the analysis, trying to find some similarities, trends among the two type of trips </h4>
# For the trip_type column 1 -> Street-hail type of trip, 2 -> Dispatch type of trip
# 
# 

# In[17]:


taxi['trip_type'].value_counts()


# In[18]:


# Visualization of the distribution of both the type of trips
sns.countplot(x='trip_type',data=taxi,palette='viridis')


# In[19]:


# At what time in tearms of hours is highest number of taxi rides took by the people.
time_hr1 = taxi['Hour_pickup'].value_counts().head(1)
print('Highest number of taxi rides requested in the particular hour : ', time_hr1)


# In[20]:


# At what time in tearms of hours is highest number of Street hail type of trip (taxi rides) took by the people.
time_hr2 = taxi[taxi['trip_type'] == 1]['Hour_pickup'].value_counts().head(1)
print('Highest number of Street hail type of trip in the particular hour : ', time_hr2)


# In[37]:


# At what time in tearms of hours is highest number of dispatch type of trip (taxi rides) took by the people.
time_hr3 = taxi[taxi['trip_type'] == 2]['Hour_pickup'].value_counts().head(1)
print('Highest number of dispatch type of trip in the particular hour : ', time_hr3)


# Which is the most busiest day for taxi rides looking at dispatch type of trip and Street hail type of trip

# In[40]:


busy_ride1= taxi[taxi['trip_type'] == 1]['Day of Week_drop'].value_counts()
print('The busiest day for Street Hail type ride is on :',busy_ride1)


# In[41]:


busy_ride2= taxi[taxi['trip_type'] == 2]['Day of Week_drop'].value_counts()
print('The busiest day for dispatch type ride is on :',busy_ride2)


# In[43]:


#countplot of the Day of Week column with the hue based off of the trip_type (dispatch & Street Hail) column.
sns.countplot(x='Day of Week_pickup',data=taxi,hue='trip_type',palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Which location is famous for the pickup for both the trip_type

# In[44]:


d3= taxi[taxi['trip_type'] == 1]['PULocationID'].value_counts().head(1)
print('The famous pickup for Street-hail type ride is on :',d3)


# In[45]:


d4= taxi[taxi['trip_type'] == 2]['PULocationID'].value_counts().head(1)
print('The famous pickup for dispatch type ride is on :',d4)


# Finding the relationship between the total_amount and tip_amount

# In[46]:


sns.lmplot(x='total_amount', y='tip_amount',hue='trip_type', data=taxi)
plt.xlabel('total_amount')
plt.ylabel('tip')
plt.title('Trend of tip with total_amount')


# In[47]:


taxi.loc[taxi['tip_amount'].idxmax()]


# The rides which are requested are for how many people and is there any different between number of people take the ride from both trip_type.

# In[48]:


d5 = taxi[taxi['trip_type'] == 1]['passenger_count'].value_counts()
d5


# In[49]:


d6 = taxi[taxi['trip_type'] == 2]['passenger_count'].value_counts()
d6


# Payment type comparision by customer 1= Credit card 2= Cash 3= No charge 4= Dispute 5= Unknown

# In[50]:


#by Street hail type
d7 = taxi[taxi['trip_type'] == 1]['payment_type'].value_counts()
d7


# In[53]:


#by dispatch type
d7 = taxi[taxi['trip_type'] == 2]['payment_type'].value_counts()
d7


# In[54]:


#countplot of the Day of Week column of the trip_type column using seaborn ploting.
sns.countplot(x='payment_type',data=taxi,hue='trip_type',palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# <h2> Predicting Model

# In[57]:


fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(taxi.corr(),cmap='coolwarm',annot=True, linewidths=1, annot_kws={"size":6})


# In[58]:


X = taxi.iloc[:,[0,3,4,5,6,7,8,9,10,11,13,14,16,17]].values
y = taxi.iloc[:,12].values


# In[59]:


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[60]:


X_train


# In[61]:


X_test


# In[62]:


y_train


# In[63]:


y_test


# In[64]:


#Creating a classification algorithm - Logistic Regression model, to predict the trip type
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[65]:


y_pred = classifier.predict(X_test)


# In[66]:


y_pred


# In[74]:


# Creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[75]:


cm


# In[69]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[70]:


#Creating a classification algorithm - Random forest classifier model, to predict the trip type
from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(random_state = 0)
classifier1.fit(X_train, y_train)


# In[71]:


y_pred1 = classifier1.predict(X_test)


# In[72]:


y_pred1


# In[76]:


from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)


# In[77]:


cm1


# In[78]:


accuracy_score(y_test, y_pred1)


# In[88]:


Y_predicted_trip_type = pd.DataFrame(y_pred1, columns=['predictions_Trip_Type'])

