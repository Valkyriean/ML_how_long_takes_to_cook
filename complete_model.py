import numpy as np
import pandas as pd
import scipy

train = pd.read_csv("data/recipe_train.csv")
npz = scipy.sparse.load_npz('data/recipe_text_features_countvec/train_steps_vec.npz')
y = train.iloc[:,-1]

tfidf_transformer  = TfidfTransformer()
X = tfidf_transformer.fit_transform(npz)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=88)

svc = svm.SVC(kernel='rbf')
svc.fit(X_train,y_train)
print("SVM Accuracy:",svc.score(X_test,y_test))
