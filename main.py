import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


import sklearn.naive_bayes as nb

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

train = pd.read_csv("data/recipe_train.csv")
test = pd.read_csv("data/recipe_test.csv")
name = pd.read_csv(r"train_name_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
ingr = pd.read_csv(r"train_ingr_doc2vec50.csv", index_col = False, delimiter = ',', header=None)
steps = pd.read_csv(r"train_steps_doc2vec50.csv", index_col = False, delimiter = ',', header=None)

X = train.iloc[:, 1:3]
y = train.iloc[:,-1]
real_x = test.iloc[:,1:3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=88)

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
