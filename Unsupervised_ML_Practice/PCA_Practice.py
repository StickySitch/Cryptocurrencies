#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hvplot.pandas
from path import Path


# In[2]:


# Loading the preprocessed iris dataset
filePath = Path('../Resources/new_iris_data.csv')
dfIris = pd.read_csv(filePath)
dfIris.head()


# In[3]:


# Standardizing data with StandardScaler
irisScaled = StandardScaler().fit_transform(dfIris)
print(irisScaled[0:5])


# In[4]:


# Initializing PCA model
pca = PCA(n_components=2)


# In[5]:


# Get two principal components for iris data.
irisPca = pca.fit_transform(irisScaled)


# In[6]:


# Transforming PCA data to a Dataframe
dfIrisPca = pd.DataFrame(
    data=irisPca,columns=['principal component 1', 'principal component 2']

)

dfIrisPca.head()


# In[7]:


# Fetching the explained variance
pca.explained_variance_ratio_


# In[8]:


# Finding best value for K
inertia = []
k = list(range(1,11))

# Calculating the inertia for the range of k values
for i in k:
    km = KMeans(n_clusters=i,random_state=0)
    km.fit(dfIrisPca)
    inertia.append(km.inertia_)

# Creaing the elbow curve
elbowData = {'k':k, 'inertia':inertia}
dfElbow = pd.DataFrame(elbowData)
dfElbow.hvplot.line(x='k', y='inertia', xticks=k,title='Elbow Curve')


# In[9]:


# Initializing the Kmeans model
model = KMeans(n_clusters=3, random_state=0)

#fitting the model
model.fit(dfIrisPca)

#predicting clusters
predict = model.predict(dfIrisPca)

# Adding the predicted class columns
dfIrisPca['class'] = model.labels_
dfIrisPca.head()


# In[10]:


dfIrisPca.hvplot.scatter(
    x='principal component 1',
    y='principal component 2',
    hover_cols=['class'],
    by='class',
)

