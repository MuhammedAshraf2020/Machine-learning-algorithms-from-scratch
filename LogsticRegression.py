import numpy as np
class LogsticRegression:
	def __init__(self , learningRate = 0.01 , itreation = 1000):
		self.learningRate = learningRate
		self.itreation    = itreation
		self.wights       = None
		self.bias         = None

	def fit(self , X , y):
		n_samples , n_feats = X.shape
		self.wights = np.zeros(n_feats)
		self.bias   = 0
		
		for _ in range(self.itreation):
			z 	        = np.dot(self.wights , X) + self.bias
			self.A      = self.sigmoid(z)
			Error       = 1  / n_samples * np.sum(- ( y * np.log(A) + (1 - y) * np.log(1 - A)))
			dz          = A - y
			dw          = 1 / n_samples  * np.sum(np.dot(X  , dz))
			db          = dz / n_samples * np.sum( dz )
			self.wights = self.wights    - self.learningRate * dw

	def sigmoid(slef , Z):
		Sig = 1 / (1+ np.exp(-Z))
		return Sig

	def _Predict(self , x):
		pred = sigmoid(np.dot(self.wights , X) + self.bias)
		return pred
	def predict(self , X):
		preds = [sample for samples in X]
		return preds

