#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[955]:


class KMeans:
    def __init__(self , n_centroids = 2 , itreation = 100):
        self.n_centroids = n_centroids
        self.itreation = itreation

    def fit(self , X ):
        samples = X.shape[0]
        self.X = X
        self.centroids = self.init_cent(self.X) 
        for i in range(self.itreation):
            idx = np.array([self.choice_groub(X[i]) for i in range(samples)])
            self.centroids = self.compute_center(X , idx)
        X_cat = []
        for i in range(self.n_centroids):
            X_cat.append(self.X[np.where(idx == i)[0]])
        return np.array(X_cat)
    def compute_center(self , X , idx):    	
        samples , n_feats = X.shape
        centroids = np.zeros((self.n_centroids , n_feats))
        for i in range(self.n_centroids):
            indecs = np.where(idx == i)
            mean_temp  = np.mean( X[indecs] , axis = 0)
            centroids[i , :] = mean_temp
        return centroids
    
    def choice_groub(self , x):
        distances = [self.eculadiean_distance(x , self.centroids[i]) for i in range(self.n_centroids)] # 
        return distances.index(min(distances))	
    
    def init_cent(self , X):
        # Function to init centroid
        samples , feats = X.shape
        cents = np.random.randint(0 , samples , self.n_centroids)
        return self.X[cents]

    def eculadiean_distance(self , first_list , second_list):
    	# function to calc eculid distance
        num = len(first_list) 
        distance = 0
        for i in range(num):
            var = pow(first_list[i] - second_list[i] , 2)
            distance = distance + var
        return np.sqrt(distance)


# In[962]:


instance = KMeans(3 , itreation = 200)


# In[927]:


age = np.random.randint(1 , 100 , 200)


# In[928]:


tal = np.random.randint(150 , 200 , 200)


# In[929]:


All = []
All.append(age)
All.append(tal)


# In[931]:


X = np.array(All).T


# In[963]:


X1 , X2 , X3  = instance.fit(X)


# In[964]:


cent = instance.centroids


# In[947]:


cent[0]


# In[965]:


plt.scatter(X1[: , 0] , X1[: , 1]  , c = "blue"   )
plt.scatter(X2[: , 0] , X2[: , 1]  , c = "red"    )
plt.scatter(X3[: , 0] , X3[: , 1]  , c = "yellow" )
#plt.scatter(X4[: , 0] , X4[: , 1]  , c = "black"  )
#plt.scatter(X5[: , 0] , X5[: , 1]  )
#plt.scatter(X6[: , 0] , X6[: , 1]   )
plt.scatter(cent[: , 0] , cent[: , 1] )

