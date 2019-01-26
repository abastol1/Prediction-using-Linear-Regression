#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Deciding whether to focus on mobile app experience or their websites.


# In[3]:


import numpy as np
import pandas as pd


# In[26]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[9]:


# Reading Ecommerce Customers csv file into DataFrame called customers

customers = pd.read_csv('Ecommerce Customers')


# In[11]:


customers.head()


# In[14]:


customers.info()


# In[15]:


customers.describe()


# In[16]:


#jointplot to compare the Time on Website and Yearly Amount Spent columns
sns.jointplot(customers['Time on Website'], customers['Yearly Amount Spent'])


# In[17]:


#jointplot to compare the Time on App and Yearly Amount Spent columns
sns.jointplot(customers['Time on App'], customers['Yearly Amount Spent'])


# In[19]:


sns.jointplot(customers['Time on App'], customers['Length of Membership'], kind='hex')


# In[20]:


sns.pairplot(customers)


# In[21]:


# Based on the above graph, Length of Membership looks to be the most 
# correlated feature with Yearly Amount Spent


# In[29]:


# Linear model plot of Yearly Amount Spent vs Length of Membership
sns.lmplot(x= 'Length of Membership', y = 'Yearly Amount Spent', data=customers)


# In[30]:


customers.columns


# In[32]:


customers.describe()


# In[33]:


# X = Numerical features of the customers
X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]


# In[35]:


# y = Yearly Amount Spent

y = customers['Yearly Amount Spent']


# In[37]:


# importing to split the data into training and testing

from sklearn.model_selection import train_test_split


# In[38]:


#Spliting the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=101 )


# In[39]:


from sklearn.linear_model import LinearRegression


# In[40]:


# Instance of LinearRegression

lm = LinearRegression()


# In[41]:


lm.fit(X_train, y_train)


# In[42]:


lm.coef_


# In[46]:


# predicting the outcome of X_test using the linear model
predictions = lm.predict(X_test)


# In[47]:


#Scatterplot of the real test values versus the predicted values

sns.scatterplot(y_test, predictions)


# In[48]:


from sklearn import metrics


# In[62]:


print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test, predictions))


# In[63]:


print("Mean Sqaure Error: ", metrics.mean_squared_error(y_test, predictions))


# In[64]:


print("Root Mean Sqaure Error: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[57]:


sns.distplot(y_test-predictions, bins=50)


# In[59]:


pd.DataFrame(lm.coef_, index = X.columns, columns=['Coeffecient'])


# In[60]:


# The linear model suggests that the company should focus more on their
# mobile app as increase in one unit time on App increases yearly spent by 26$
# but increase in one unit time on website only increases 0.2$


# In[ ]:




