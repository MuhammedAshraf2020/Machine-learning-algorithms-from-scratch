import numpy as np

class LinearRegression:
    
    def __init__(self , learning_rate = 0.01 , n_itres = 1000 , norm = False ):
        self.learning_rate = learning_rate
        self.n_itres       = n_itres
        self.weights       = None
        self.bias          = None 
        self.norm          = norm

    def fit(self , X , y):
        if self.norm == True:
            self.NormalEquation(X , y)
        else:
            self.GradinetDescent(X , y)

    def NormalEquation(self , X , y):
        n        = len(X)
        X        = np.concatenate((np.ones((n , 1)) , X) , axis = 1)
        XT       = X.transpose()
        XTX      = np.dot(XT , X)
        XTXinv   = np.linalg.inv(XTX)
        XTXinvXT = np.dot(XTXinv , XT)
        self.B   = np.dot(XTXinvXT , y)
    
    def GradinetDescent(self , X , y):
        self.weights = np.zeros(n_feats)
        self.bias    = 0

        for _ in range(self.n_itres):
            y_predicted = np.dot(X , self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T , (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights = self.weights - self.learning_rate * dw
            self.bias    = self.bias - self.learning_rate * db

    def _predictNorm(self , x_test):
        x_test = np.concatenate(([1] , x_test ))
        return np.dot(x_test , self.B)        
    
    def _predictGD(self , X):
        y_predicted = np.dot(X , self.weights) + self.bias
        return y_predicted

    def predict(self , x_test):
        if self.norm == True:
            _predictNorm(x_test)
        else:
            _predictGD(x_test)