import numpy as np
import pandas as pd
import scipy
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sklearn.naive_bayes as nb
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer


train = pd.read_csv("data/recipe_train.csv")
test = pd.read_csv("data/recipe_test.csv")
name = pd.read_csv(r"data/recipe_text_features_doc2vec50/train_name_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
ingr = pd.read_csv(r"data/recipe_text_features_doc2vec50/train_ingr_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
steps = pd.read_csv(r"data/recipe_text_features_doc2vec50/train_steps_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
real_name = pd.read_csv(r"data/recipe_text_features_doc2vec50/test_name_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
real_ingr = pd.read_csv(r"data/recipe_text_features_doc2vec50/test_ingr_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
real_steps = pd.read_csv(r"data/recipe_text_features_doc2vec50/test_steps_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
vocab = pickle.load(open("data/recipe_text_features_countvec/train_steps_countvectorizer.pkl", "rb"))
npz = scipy.sparse.load_npz('data/recipe_text_features_countvec/train_steps_vec.npz')
t_npz = scipy.sparse.load_npz('data/recipe_text_features_countvec/test_steps_vec.npz')
vocab_dict = vocab.vocabulary_
# X = train.iloc[:, 3:4]
y = train.iloc[:,-1]
# X = pd.concat([X,steps,name,ingr], axis = 1)
# real_x = test.iloc[:,1:3]
# real_x = pd.concat([real_x, real_steps], axis=1)
tfidf_transformer  = TfidfTransformer()
X = tfidf_transformer.fit_transform(npz)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=88)
# real_x = tfidf_transformer.fit_transform(t_npz)

# x2 = SelectKBest(chi2, k=5)
# X_train_x2 = x2.fit_transform(X_train,y_train)
# X_test_x2 = x2.transform(X_test)

# mi = SelectKBest(score_func=mutual_info_classif, k=10)
# X_train_mi = mi.fit_transform(X_train,y_train)
# X_test_mi = mi.transform(X_test)

# lgr = LogisticRegression()
# lgr.fit(X,y)

svc = svm.SVC()
svc.fit(X_train,y_train)

# print("lgr Accuracy:",svc.score(X_test,y_test))

y = svc.predict(X_test)
# real_y = svc.predict(real_x)
# index = range(1,len(real_y)+1)
out = pd.DataFrame(data = y)
out.to_csv('out.csv')