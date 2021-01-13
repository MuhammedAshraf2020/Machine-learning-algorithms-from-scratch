def train_test_split(X , y , rate = 0.60):
    X_temp   =  X
    Y_temp   =  y
    num      = len(X)
    Rate_Num = int(num * rate)
    Test     = np.random.randint(0 , num , Rate_Num)
    x_train  = [X_temp.pop(i) for i in Test]
    y_train  = [Y_temp.pop(i) for i in Test]
    return x_train , X_temp , y_train , Y_temp


def accuracy(pridect , y_test , cl = False):
    List = []
    n    = len(pridect)
    if cl == False:
        for i in range(n):
            temp = float(abs(pridect[i] - y_test[i]))
            Temp = (temp / float(y_train[i])) * 100
            List.append(float(temp))
        return sum(List) / n
    elif cl==True:
        counter = 0
        for i in range(n):
            if pridect[i] != y_test[i]:
                counter +=1
        return counter / n *100
