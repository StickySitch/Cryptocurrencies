#!/usr/bin/env python
# coding: utf-8

# # Clustering Crypto

# In[27]:


# Initial imports
import pandas as pd
import hvplot.pandas
from path import Path
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


# ### Deliverable 1: Preprocessing the Data for PCA

# In[28]:


# Loading the crypto_data.csv dataset.
filePath = Path('Resources/crypto_data.csv')
cryptoDf = pd.read_csv(filePath, index_col=0)
cryptoDf.head()


# In[29]:


# Checking Value type for each column
cryptoDf.dtypes


# In[30]:


# Keeping all the cryptocurrencies that are being traded by filtering the data frame
cryptoTradingDf = cryptoDf[cryptoDf['IsTrading'] == True]
cryptoTradingDf


# In[31]:


# Looking for columns with null values
cryptoTradingDf.isnull().value_counts()


# In[32]:


# Removing the "IsTrading column.
cryptoTradingDf.drop(columns=['IsTrading'], axis=1, inplace=True)
cryptoTradingDf.head()


# In[33]:


# Remove rows that have at least 1 null value.
cryptoTradingDf.dropna(inplace=True)
cryptoTradingDf


# In[34]:


# Keeping the rows where coins are mined.
cryptoTradingDf = cryptoTradingDf[cryptoTradingDf['TotalCoinsMined'] > 0]
cryptoTradingDf


# In[35]:


# Creating a new DataFrame that holds only the cryptocurrencies names.
cryptoNameDf = cryptoTradingDf['CoinName']
cryptoNameDf


# In[36]:


# Dropping the 'CoinName' column since it's not going to be used on the clustering algorithm.
cryptoTradingDf.drop(columns=['CoinName'], axis=1, inplace=True)
cryptoTradingDf.head()


# In[37]:


# Using get_dummies() to create variables for text features.


cryptoTradingDfEncoded = cryptoTradingDf.copy()


X = pd.get_dummies(cryptoTradingDfEncoded, columns=['Algorithm', 'ProofType'])
X.head()


# In[38]:


# Standardize the data with StandardScaler().
cryptoTradingScaled = StandardScaler().fit_transform(X)

print(cryptoTradingScaled[0:5])


# ### Deliverable 2: Reducing Data Dimensions Using PCA

# In[39]:


# Using PCA to reduce dimension to three principal components.
indexList = (X.index.to_list())
# Initializing PCA model
pca = PCA(n_components=3)

# getting 3 principal components
cryptoPca = pca.fit_transform(cryptoTradingScaled)

# Transforming PCA data to a Dataframe
cryptoPcaDf = pd.DataFrame(
    data=cryptoPca,
    columns=['principal component 1', 'principal component 2', 'principal component 3'],
    index=indexList

)

cryptoPcaDf


# ### Deliverable 3: Clustering Crytocurrencies Using K-Means
# 
# #### Finding the Best Value for `k` Using the Elbow Curve

# In[40]:


# Initializing inertia list and Instantiating a list of K values to test with
inertia = []
k = list(range(1,11))

# Looking for best K value
for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(cryptoPcaDf)
    inertia.append(km.inertia_)


# In[41]:


# Creating an elbow curve to find the best value for K.
elbowData = {'k':k, 'inertia': inertia}
dfElbow = pd.DataFrame(elbowData)
dfElbow.hvplot.line(x='k', y='inertia', title='Crypto Elbow Curve', xticks=k)


# Running K-Means with `k=4`

# In[42]:


# Initializing the K-Means model
model = KMeans(n_clusters=4, random_state=0)

# fitting the model
model.fit(cryptoPcaDf)

# Predicting K clusters
predictions = model.predict(cryptoPcaDf)
print(predictions)


# Adding class column to df
cryptoPcaDf['Class'] = model.labels_


# In[43]:


# Creating a new DataFrame including predicted clusters and cryptocurrencies features.
# Concatentating the crypto_df and pcs_df DataFrames on the same columns.
clustered_df = cryptoTradingDf.join(cryptoPcaDf,how='inner')
clustered_df.head()


# In[44]:


#  Adding a new column, "CoinName" to the clustered_df DataFrame that holds the names of the cryptocurrencies.
clustered_df = clustered_df.join(cryptoNameDf,how='inner')
clustered_df.head()


# In[45]:


# Print the shape of the clustered_df
print(clustered_df.shape)
clustered_df.head(10)


# ### Deliverable 4: Visualizing Cryptocurrencies Results
# 
# #### 3D-Scatter with Clusters

# In[46]:


# Creating a 3D-Scatter with the PCA data and the clusters

fig = px.scatter_3d(
    clustered_df,
    x='principal component 1',
    y='principal component 2',
    z='principal component 3',
    color='Class',
    symbol='Class',
    hover_name='CoinName',
    hover_data=['TotalCoinsMined','TotalCoinSupply','Algorithm', 'ProofType']
)

fig.update_layout(legend=dict(x=0, y=1))
fig.show()


# In[47]:


# Creating a table with tradable cryptocurrencies.
clustered_df.hvplot.table(columns=['CoinName','Algorithm','ProofType','TotalCoinSupply','TotalCoinsMined','Class'])


# In[48]:


# Print the total number of tradable cryptocurrencies.
clustered_df['CoinName'].count()


# In[49]:


# Scaling data to create the scatter plot with tradable cryptocurrencies.
clustDf = clustered_df[['TotalCoinSupply','TotalCoinsMined']]
minMax = MinMaxScaler().fit_transform(clustDf)
minMax


# In[50]:


# Create a new DataFrame that has the scaled data with the clustered_df DataFrame index.
scatterDf = pd.DataFrame(
    data=minMax,
    columns=['TotalCoinSupplyScaled', 'TotalCoinsMinedScaled'],
    index= clustered_df.index.tolist()
)
scatterDf.head()

# Adding the "CoinName" column from the clustered_df DataFrame to the new DataFrame.
scatterDf = scatterDf.join(cryptoNameDf,how='inner')

# Adding the "Class" column from the clustered_df DataFrame to the new DataFrame.
classColumn = clustered_df['Class']
scatterDf = scatterDf.join(classColumn, how='inner')

scatterDf.head(10)


# In[51]:


# Create a hvplot.scatter plot using x="TotalCoinsMined" and y="TotalCoinSupply".
scatterDf.hvplot.scatter(x='TotalCoinsMinedScaled',
                         y='TotalCoinSupplyScaled',
                         by='Class',
                         xlabel='Total Coins Mined',
                         ylabel='Total Coin Supply',
                         title='Crypto Scatter',
                         hover_cols=['CoinName']
                         )


# In[51]:




