import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train = pd.read_csv("data/recipe_train.csv")
test = pd.read_csv("data/recipe_test.csv")

X = train.iloc[:, 1:3]
y = train.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=88)
lgr = LogisticRegression()

lgr.fit(X_train,y_train)
print("lgr Accuracy:",lgr.score(X_test,y_test))
