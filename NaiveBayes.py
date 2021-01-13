import numpy as np
from collections import Counter
class GausianNB:
    def fit(self , X , y):
        self.classes   = np.unique(y)
        n_classes = len(self.classes)
        n_feature      = X.shape[1]
        n_samples      = len(X)
        self.List_mean =  np.zeros((n_classes , n_feature))
        self.List_var  =  np.zeros((n_classes , n_feature))
        self.Priors    =  np.zeros((n_classes))
        index = 0
        for i in self.classes:
            X_C = X[i == y]
            self.List_mean[index , : ]  = X_C.mean(axis = 0)
            self.List_var[index  , : ]  = X_C.mean(axis = 0)
            self.Priors[index] = len(X_C) / n_samples
            index+=1
    
    def predict(self , X):
        pred = [self._predict(i) for i in X]
        return pred
    
    def _predict(self , x):
        postrious = []
        for i , j in enumerate(self.classes):
            Prior = np.log(self.Priors[i])
            post  = sum(np.log(self.N_Dist(x , i)))
            postrious.append(Prior + post)
        return self.classes[np.argmax(postrious)]
    def N_Dist(self , z , i):
        Mean   = self.List_mean[i]
        var    = self.List_var[i]
        First  = np.sqrt(np.pi*var)
        Second = np.exp(-(z - Mean)**2 / 2 * var)
        return First / Second