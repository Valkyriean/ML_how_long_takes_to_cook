import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv("data/recipe_train.csv")
X = train.iloc[:, 3:4]
y = train.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=88)
predict = pd.read_csv("out.csv", header = None)
predict = predict.iloc[:,1]


c_mat = [[0,0,0],[0,0,0],[0,0,0]]

# 13200
for i in range(13200):
    y_predict = predict.iloc[i]
    real = y_test.iloc[i]
    y_predict = int(y_predict) -1
    real=int(real) -1
    c_mat[real][y_predict] +=1
print(c_mat)

# precision = tp/(tp+fp)
# recall = tp/(tp+fn)

tp1 = c_mat[0][0]
tp2 = c_mat[1][1]
tp3 = c_mat[2][2]

fn1 = c_mat[0][1] + c_mat[0][2]
fn2 = c_mat[1][0] + c_mat[1][2]
fn3 = c_mat[2][0] + c_mat[2][1]

fp1 = c_mat[1][0] + c_mat[2][0]
fp2 = c_mat[0][1] + c_mat[2][1]
fp3 = c_mat[0][2] + c_mat[1][2]

pre1 = tp1/(tp1 + fp1)
pre2 = tp2/(tp2 + fp2)
pre2 = tp3/(tp3 + fp3)

rec1 = tp1/(tp1+fn1)
rec1 = tp2/(tp2+fn2)
rec1 = tp3/(tp3+fn3)