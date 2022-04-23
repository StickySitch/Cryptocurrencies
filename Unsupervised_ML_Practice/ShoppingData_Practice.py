#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing dependencies
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import hvplot.pandas


# In[2]:


# Load data
filePath = "../Resources/shopping_data_cleaned.csv"
dfShopping = pd.read_csv(filePath)
dfShopping.head(10)


# In[3]:


# Calculate the inertia for the range of K values
inertia = []
k = list(range(1,11))

for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(dfShopping)
    inertia.append(km.inertia_)


# In[4]:


# Creating elbow curve for shopping data
elbowData = {'k':k, 'inertia':inertia}
dfElbow = pd.DataFrame(elbowData)
dfElbow.hvplot.line(x='k', y='inertia',xticks=k, title="Elbow Curve")


# In[5]:


def get_clusters(k, data):
    # Create a copy of the DataFrame
    data = data.copy()

    # Initialize the K-Means model
    model = KMeans(n_clusters=k, random_state=0)

    # Fit the model
    model.fit(data)

    # Predict clusters
    predictions = model.predict(data)

    # Create return DataFrame with predicted clusters
    data["class"] = model.labels_

    return data


# In[6]:


# Creating and fitting the model with 5 K values
fiveClusters = get_clusters(5, dfShopping)
fiveClusters.head()


# In[7]:


# Creating and fitting the model with 6 K values
sixClusters = get_clusters(6, dfShopping)
sixClusters.head()


# In[8]:


# Plotting the 5 cluster model
fiveClusters.hvplot.scatter(x='Annual_Income', y='Spending_Score', by='class')


# In[9]:


# 3D Plotting 5 cluster model
fig = px.scatter_3d(
    fiveClusters,
    x="Age",
    y="Spending_Score",
    z="Annual_Income",
    color="class",
    symbol="class",
    width=800,
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()


# In[10]:


# Plotting the 6 cluster model
sixClusters.hvplot.scatter(x='Annual_Income', y='Spending_Score', by='class')


# In[11]:


# 3D Plotting 6 cluster model
fig = px.scatter_3d(
    sixClusters,
    x="Age",
    y="Spending_Score",
    z="Annual_Income",
    color="class",
    symbol="class",
    width=800,
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()

