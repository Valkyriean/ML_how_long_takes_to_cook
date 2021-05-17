import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder




train = pd.read_csv("data/recipe_train.csv")
test = pd.read_csv("data/recipe_test.csv")
X = train.iloc[:, 1:3]
y = train.iloc[:,-1]
real_x = test.iloc[:,1:3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=88)

lgr = LogisticRegression()
lgr.fit(X_train,y_train)
real_y = lgr.predict(real_x)
print("Accuracy:",lgr.score(X_test,y_test))
index = range(1,len(real_y)+1)
out = pd.DataFrame(data = real_y, index= index)
out.to_csv('out.csv', header = ['duration_label'], index = True, index_label='id')
