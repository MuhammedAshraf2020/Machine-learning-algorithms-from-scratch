import numpy as np
class KNN:
    def __init__(self , k = 3):
        self.k = 3

    def eculadiean_distance(self , first_list , second_list):
        self.first_list  = first_list
        self.second_list = second_list
        num = len(self.first_list) # numper of element(features)
        self.distance = 0
        for i in range(num):
            var = pow(self.first_list[i] - self.second_list[i] , 2)
            self.distance = self.distance + var
        return sqrt(self.distance)
    
    def sort2(self, T):
        self.T = T
        x = T
        for item in  range(len(x)):
            for i in range(len(x) - 1):
                    if x[i]>x[i+1]:
                        stable = x[i+1]
                        x[i+1] = x[i]
                        x[i] = stable
        return x
    def get_index(self , item , array):
        self.item = item
        self.array= array
        for i in range(len(self.array)):
            if self.array[i] == self.item:
                return i

    def get_common(self , arr):
        self.arr = arr
        count = 0
        rank  = []
        for i in range(len(self.arr)):
            for j in range(len(self.arr)):
                if self.arr[i] == self.arr[j]:
                    count+=1
            rank.append(count)
            count = 0
        rank_2  = self.sort2(rank)
        var   = rank_2[-1]
        var_2 = self.get_index(var , rank)
        return self.arr[var_2]
    
    def _predection(self , x ):
        self.x = x
        self.distances = [self.eculadiean_distance(x , self.x_train[i]) for i in range(len(self.x_train))]
        return self.distances

    def predict(self , x):
        self.x = x
        indecies = [self.get_index(self.sort2(self._predection(x))[i] , self._predection(x)) for i in range(self.k)]
        target = [self.y_train[i] for i in indecies]
        nn = self.get_common(target)
        return nn
