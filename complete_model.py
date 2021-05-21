import numpy as np
import pandas as pd
import scipy
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

train = pd.read_csv("recipe_train.csv")
npz = scipy.sparse.load_npz('train_steps_vec.npz')
y = train.iloc[:,-1]

tfidf_transformer  = TfidfTransformer()
X = tfidf_transformer.fit_transform(npz)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=88)


# mi = SelectKBest(score_func=mutual_info_classif, k=10)
# X_train_mi = mi.fit_transform(X_train,y_train)
# X_test_mi = mi.transform(X_test)

svc = svm.SVC(kernel='rbf')
svc.fit(X_train,y_train)
print("SVM Accuracy:",svc.score(X_test,y_test))

# Predict for test
# t_npz = scipy.sparse.load_npz('test_steps_vec.npz')
# real_x = tfidf_transformer.fit_transform(t_npz)
# real_y = SVC.predict(real_x)
# index = range(1,len(real_y)+1)
# out = pd.DataFrame(data = real_y, index= index)
# out.to_csv('out.csv', header = ['duration_label'], index = True, index_label='id')