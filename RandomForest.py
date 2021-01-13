import numpy as np
from collections import Counter
from Tree import DecisionTree
class RandomForest:
    def __init__(self , n_trees = 10 , min_samples_split = 2 , max_depth = 100 , n_feats = None):
        self.n_feats           = n_feats
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split 
        self.n_trees           = n_trees

    def fit(self , x , y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split = self.min_samples_split , 
                max_depth = self.max_depth , n_feats = self.n_feats)
            X_samp , y_samp = self.bootstrap_sample(x , y)
            tree.fit(X_samp , y_samp)
            self.trees.append(tree)
    def most_common_label( self , y):
        counter = Counter(y)
        print(counter)
        most_common_label = counter.most_common(1)[0][0]
        return most_common_label
    
    def bootstrap_sample(self , x , y):
        n_sampels = x.shape[0]
        idxs = np.random.choice(n_sampels , n_sampels , replace = True)
        return x[idxs] , y[idxs]

    def predict(self , X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds , 0 , 1)
        y_pred     = [self.most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)