#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from path import Path


# # Pandas Refresher

# In[2]:


filePath = Path('../Resources/iris.csv')
irisDf = pd.read_csv(filePath)
irisDf.head()


# In[3]:


irisDf = irisDf.drop(['class'], axis=1)
irisDf.head()


# In[4]:


irisDf = irisDf[['sepal_length', 'petal_length', 'sepal_width', 'petal_width']]
irisDf.head()


# In[5]:


OutputPath = 'Resources/new_iris_data.csv'
irisDf.to_csv(OutputPath, index=False)


# # Shopping Data Preprocessing

# In[ ]:


# Loading the shopping data from "shopping_data.csv"
filePath = Path('../Resources/shopping_data.csv')
shoppingDf = pd.read_csv(filePath, encoding='ISO-8859-1')
shoppingDf.head()


# In[ ]:


# Checking Columns
shoppingDf.columns


# In[ ]:


# Checking column data types
shoppingDf.dtypes


# In[ ]:


# Find null values
for column in shoppingDf.columns:
    print(f"Column {column} has {shoppingDf[column].isnull().sum()} null values")


# In[ ]:


# Dropping null rows
shoppingDf = shoppingDf.dropna()


# In[ ]:


# Checking for duplicates
print(f'Duplicate Entries {shoppingDf.duplicated().sum()}')


# In[ ]:


# Removing customer ID column
shoppingDf.drop(columns=['CustomerID'], inplace=True)
shoppingDf.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Encoding Card Member column (1=Yes, 0=No)
le = LabelEncoder()

shoppingDfEncoded = shoppingDf.copy()

shoppingDfEncoded['Card Member'] = le.fit_transform(shoppingDfEncoded['Card Member'])

shoppingDfEncoded.head()


# In[ ]:


# Scaling down the Annual income column
shoppingDfEncoded['Annual Income'] = shoppingDfEncoded['Annual Income'] / 1000


shoppingDfEncoded.columns = shoppingDfEncoded.columns.str.replace(' ', '_')

shoppingDfEncoded.rename(columns={'Spending_Score_(1-100)': 'Spending_Score'}, inplace=True)

shoppingDfEncoded.head()


# In[ ]:


# Saving Cleaned Data
file_path = '../Resources/shopping_data_cleaned.csv'
shoppingDfEncoded.to_csv(file_path, index=False)


# # KMeans

# In[ ]:


import pandas as pd
import plotly.express as px
import hvplot.pandas
from sklearn.cluster import KMeans
# Initializing model with K = 3 (Since we already know there are three classes of iris plants)
model = KMeans(n_clusters=3,random_state=5)
model


# In[ ]:


# Fitting the model
model.fit(irisDf)


# In[ ]:


predict = model.predict(irisDf)
print(predict)


# In[ ]:


# Add a new class column to the df_iris
irisDf['class'] = model.labels_
irisDf.head()


# In[ ]:


import plotly.express as px
import hvplot.pandas

# Plotting the clusters with two features
irisDf.hvplot.scatter(x='sepal_length',y='sepal_width', by='class')


# In[ ]:


# plotting the clusters with three features (3D)
fig = px.scatter_3d(
    irisDf,
    x='petal_width',
    y='sepal_length',
    z='petal_length',
    color='class',
    symbol='class',
    size='sepal_width',
    width=800,
)
fig.update_layout(legend=dict(x=0,y=1))
fig.show()


# In[ ]:




