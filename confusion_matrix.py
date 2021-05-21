import numpy as np
import pandas as pd
import scipy
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer

train = pd.read_csv("recipe_train.csv")
npz = scipy.sparse.load_npz('train_steps_vec.npz')
y = train.iloc[:,-1]

tfidf_transformer  = TfidfTransformer()
X = tfidf_transformer.fit_transform(npz)

y = train.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)


svc = svm.SVC(kernel='rbf')
svc.fit(X_train,y_train)
predict = svc.predict(X_test)
c_mat = [[0,0,0],[0,0,0],[0,0,0]]

for i in range(8000):
    y_predict = predict[i]
    real = y_test.iloc[i]
    y_predict = int(y_predict) -1
    real=int(real) -1
    c_mat[real][y_predict] +=1
print(c_mat)

# precision = tp/(tp+fp)
# recall = tp/(tp+fn)

# c_mat = [[2825, 687, 9], [799, 3263, 18], [74, 101, 224]]

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
pre3 = tp3/(tp3 + fp3)

rec1 = tp1/(tp1+fn1)
rec2 = tp2/(tp2+fn2)
rec3 = tp3/(tp3+fn3)

print(pre1)
print(pre2)
print(pre3)
print(rec1)
print(rec2)
print(rec3)