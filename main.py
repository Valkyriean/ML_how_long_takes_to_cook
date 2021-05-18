import numpy as np
import pandas as pd
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


train = pd.read_csv("data/recipe_train.csv")
test = pd.read_csv("data/recipe_test.csv")
name = pd.read_csv(r"data/recipe_text_features_doc2vec50/train_name_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
ingr = pd.read_csv(r"data/recipe_text_features_doc2vec50/train_ingr_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
steps = pd.read_csv(r"data/recipe_text_features_doc2vec50/train_steps_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
real_name = pd.read_csv(r"data/recipe_text_features_doc2vec50/test_name_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
real_ingr = pd.read_csv(r"data/recipe_text_features_doc2vec50/test_ingr_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
real_steps = pd.read_csv(r"data/recipe_text_features_doc2vec50/test_steps_doc2vec50.csv", index_col = False, delimiter = ',', header=None)


X = train.iloc[:, 1:3]
y = train.iloc[:,-1]
X = pd.concat([X,steps,name,ingr], axis = 1)
real_x = test.iloc[:,1:3]
real_x = pd.concat([real_x, real_steps], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=88)


# x2 = SelectKBest(chi2, k=5)
# X_train_x2 = x2.fit_transform(X_train,y_train)
# X_test_x2 = x2.transform(X_test)

# mi = SelectKBest(score_func=mutual_info_classif, k=10)
# X_train_mi = mi.fit_transform(X_train,y_train)
# X_test_mi = mi.transform(X_test)

# lgr =  MLPClassifier(random_state=1, max_iter=300)
lgr = LogisticRegression()
lgr.fit(X_train,y_train)
print("lgr Accuracy:",lgr.score(X_test,y_test))

# real_y = lgr.predict(real_x)
# index = range(1,len(real_y)+1)
# out = pd.DataFrame(data = real_y, index= index)
# out.to_csv('out.csv', header = ['duration_label'], index = True, index_label='id')



# gnb = GaussianNB()
# mnb = MultinomialNB()
# bnb = BernoulliNB()

# gnb.fit(X_train, y_train)
# acc = gnb.score(X_test, y_test)
# print("GNB score %f " %acc)
    
# mnb.fit(X_train, y_train)
# acc = mnb.score(X_test, y_test)
# print("MNB score %f " %acc)
    
# bnb.fit(X_train, y_train)
# acc = bnb.score(X_test, y_test)
# print("BNB score %f " %acc)
