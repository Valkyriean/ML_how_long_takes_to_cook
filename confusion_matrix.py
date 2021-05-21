import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv("data/recipe_train.csv")
X = train.iloc[:, 3:4]
y = train.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=88)
predict = pd.read_csv("out.csv", header = None)
predict = predict.iloc[:,1]


mat = [[0,0,0],[0,0,0],[0,0,0]]

# 13200
for i in range(13200):
    y_predict = predict.iloc[i]
    real = y_test.iloc[i]
    y_predict = int(y_predict) -1
    real=int(real) -1
    mat[real][y_predict] +=1
print(mat)
