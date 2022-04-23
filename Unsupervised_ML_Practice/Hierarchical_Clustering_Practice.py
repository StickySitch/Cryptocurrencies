#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import hvplot.pandas
from path import Path


# In[2]:


filePath = Path('../Resources/new_iris_data.csv')
dfIris = pd.read_csv(filePath)
dfIris.head()


# In[3]:


irisScaled = StandardScaler().fit_transform(dfIris)
print(irisScaled[0:5])


# In[4]:


pca = PCA(n_components=2)

irisPca = pca.fit_transform(irisScaled)


# In[5]:


dfIrisPca = pd.DataFrame(
    data=irisPca,columns=['principal component 1', 'principal component 2']
)
dfIrisPca.head()


# In[6]:


import plotly.figure_factory as ff

# Creating the dendrogram
fig = ff.create_dendrogram(dfIrisPca,color_threshold=0)
fig.update_layout(width=800, height=500)
fig.show()


# In[7]:


agg = AgglomerativeClustering(n_clusters=3)
model = agg.fit(dfIrisPca)


# In[8]:


dfIrisPca['class'] = model.labels_
dfIrisPca.head()


# In[9]:


dfIrisPca.hvplot.scatter(
    x='principal component 1',
    y='principal component 2',
    hover_cols=['class'],
    by='class',
)


# In[9]:




