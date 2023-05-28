## Task-1 Import libraries

import pandas as pd 
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.decomposition import PCA

## Importing the dataset
creditcard_df=pd.read_csv('CC GENERAL.csv')

## Visualize and explore the data set

# Let's see if we have any missing data

sns.heatmap(creditcard_df.isnull(),yticklabels=False,cbar=False,cmap='Blues') ## so mapping the missing data
# plt.show()

print(creditcard_df.isnull().sum())


# As the nullvalues are only present in Min payment colunm\
# Filling up the missing elements with mean of the 'Minimun Paymennt'
creditcard_df.loc[(creditcard_df['MINIMUM_PAYMENTS'].isnull()==True),'MINIMUM_PAYMENTS']=creditcard_df['MINIMUM_PAYMENTS'].mean()
creditcard_df.loc[(creditcard_df['CREDIT_LIMIT'].isnull()==True),'CREDIT_LIMIT']=creditcard_df['CREDIT_LIMIT'].mean()

# Checking for any dduplicated entries in the data
print('Duplicate entries in the data are: ',creditcard_df.duplicated().sum())

# Drop the 'CUST_ID' column
creditcard_df.drop('CUST_ID',axis=1,inplace=True)
print(creditcard_df.columns)


## displot combines the matplotlib.hist fun with seaborn kdeplot()

plt.figure(figsize=(10,50))
for i in range(len(creditcard_df.columns)):
  plt.subplot(17, 1, i+1)
  sns.distplot(creditcard_df[creditcard_df.columns[i]], kde_kws={"color": "b", "lw": 3, "label": "KDE"}, hist_kws={"color": "g"})
  plt.title(creditcard_df.columns[i])

plt.tight_layout()

plt.show()


## Find the optimal numr of cluster using elw method

#Lets scale the data  first
scaler=StandardScaler()
creditcard_df_scaled=scaler.fit_transform(creditcard_df)

print(creditcard_df_scaled)

# Applying K-means method

kmeans = KMeans(7)
kmeans.fit(creditcard_df_scaled)
labels = kmeans.labels_


cluster_centers=pd.DataFrame(data=kmeans.cluster_centers_,columns=[creditcard_df.columns])


# In order to understand what these numbers mean, let's perform inverse transformation
cluster_centers=scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [creditcard_df.columns])
y_kmeans = kmeans.fit_predict(creditcard_df_scaled)
y_kmeans

# concatenate the clusters labels to our original dataframe
creditcard_df_cluster = pd.concat([creditcard_df, pd.DataFrame({'cluster':labels})], axis = 1)
creditcard_df_cluster.head()
# Plot the histogram of various clusters
for i in creditcard_df.columns:
  plt.figure(figsize = (35, 5))
  for j in range(8):
    plt.subplot(1,8,j+1)
    cluster = creditcard_df_cluster[creditcard_df_cluster['cluster'] == j]
    cluster[i].hist(bins = 20)
    plt.title('{}    \nCluster {} '.format(i,j))
  
  plt.show()

## priciple component analyisis and visuvalize
# Obtain the principal components 
pca=PCA(n_components=2)
principal_comp=pca.fit_transform(creditcard_df_scaled)
principal_comp


# Create a dataframe with the two components

pca_df = pd.DataFrame(data = principal_comp, columns =['pca1','pca2'])
pca_df.head()


# Concatenate the clusters labels to the dataframe
pca_df = pd.concat([pca_df,pd.DataFrame({'cluster':labels})], axis = 1)
pca_df.head()

plt.figure(figsize=(10,10))
ax = sns.scatterplot(x="pca1", y="pca2", hue = "cluster", data = pca_df, palette =['red','green','blue','pink','yellow','gray','purple','black'])
plt.show()



## Completed

# Chalange
correlations = creditcard_df.corr()
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(correlations, annot = True)
plt.show()

# 'PURCHASES' have high correlation between one-off purchases, 'installment purchases, purchase transactions, credit limit and payments. 
# Strong Positive Correlation between 'PURCHASES_FREQUENCY' and 'PURCHASES_INSTALLMENT_FREQUENCY'
