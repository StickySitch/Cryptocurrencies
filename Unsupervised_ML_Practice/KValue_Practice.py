#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Dependencies
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import hvplot.pandas
from path import Path


# In[2]:


# Loading in cleaned shopping data
filePath = Path('../Resources/shopping_data_cleaned.csv')
shoppingDf = pd.read_csv(filePath)
shoppingDf.head()


# In[3]:


# Plotting
shoppingDf.hvplot.scatter(x='Annual_Income', y='Spending_Score')


# In[4]:


# Function to cluster and plot dataset
def testClusterAmount(df, clusters):
    model = KMeans(n_clusters=clusters, random_state=5)
    model

    #fitting the model
    model.fit(df)

    # Adding a new class column to df
    df['class'] = model.labels_


# In[5]:


# Creating 2 cluster model

testClusterAmount(shoppingDf, 2)
shoppingDf.hvplot.scatter(x='Annual_Income', y='Spending_Score', by='class')


# In[6]:


# Creating 3D plot for a better view of the clusters
fig = px.scatter_3d(
    shoppingDf,
    x='Annual_Income',
    y='Spending_Score',
    z='Age',
    color='class',
    symbol='class',
    width=800,

)

fig.update_layout(legend=dict(x=0, y=1))
fig.show()


# In[7]:


# Creating 3 cluster model

testClusterAmount(shoppingDf, 3)
shoppingDf.hvplot.scatter(x='Annual_Income', y='Spending_Score', by='class')


# In[8]:


# Creating 3D plot for a better view of the clusters
fig = px.scatter_3d(
    shoppingDf,
    x='Annual_Income',
    y='Spending_Score',
    z='Age',
    color='class',
    symbol='class',
    width=800,

)

fig.update_layout(legend=dict(x=0, y=1))
fig.show()


# # Iris Elbow Curve

# In[11]:


# Importing dependencies
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import hvplot.pandas


# In[12]:


# Loading data
file_path = "../Resources/new_iris_data.csv"
df_iris = pd.read_csv(file_path)

df_iris.head(10)


# In[13]:


# Initializing inertia list and Instantiatings a list of K values to test with
inertia = []
k = list(range(1,11))


# In[14]:


# Looking for the best K
for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(df_iris)
    inertia.append(km.inertia_)


# In[15]:


# Defining a dataframe to plot the elbow curve using hvplot
elbowData = {'k': k, 'inertia': inertia}
dfElbow = pd.DataFrame(elbowData)
dfElbow.hvplot.line(x='k',y='inertia', title='Elbow Curve', xticks=k)


# In[ ]:




