import pandas as pd
import numpy as np
data_raw = pd.read_csv("./titanic/train.csv")
data_test = pd.read_csv("./titanic/test.csv")
data_train = data_raw.copy(deep=True)
data_clean = [data_train, data_test]
data_train.head() 
data_train.tail()  
data_train.sample(5) 
data_train.info()
data_train.describe(include="all")
data_test.head()
data_test.tail()
data_test.sample(5)
data_test.info()
data_test.describe(include="all")
data_train.isnull().sum()
print("训练集各特征缺失率：\n", data_train.isnull().sum()/data_train.shape[0])
data_test.isnull().sum()
print("测试集各特征的缺失率：\n", data_test.isnull().sum()/data_test.shape[0])
data_train["Survived"].value_counts()
print("存活率：\n", data_train["Survived"].value_counts()/data_train.shape[0])
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.rcParams["font.sans-serif"] = ["SimHei"]
data_train.info()
for dataset in data_clean:
    dataset.drop(columns=["PassengerId", "Ticket"], inplace=True)
data_train.info()
data_test.info()
data_raw["Cabin"].value_counts()
for dataset in data_clean:
    dataset.drop(columns=["Cabin"], inplace=True)
data_train.info()
data_test.info()
fig = plt.figure(figsize=(20, 8))
plt.suptitle("定类/定序分析：Embarked, Pclass, Sex")  # 大标题
plt.subplot(131)  
sns.countplot(x="Embarked", hue="Survived", data=data_train)
plt.subplot(132)
sns.countplot(x="Pclass", hue="Survived", data=data_train)
plt.subplot(133)
sns.countplot(x="Sex", hue="Survived", data=data_train)
plt.show()
data_train[["Embarked", "Survived"]].groupby(["Embarked"]).mean()
data_train[["Pclass", "Survived"]].groupby(["Pclass"]).mean()
data_train[["Sex", "Survived"]].groupby(["Sex"]).mean()

fig = plt.figure(figsize=(20, 8))
plt.suptitle("定类/定序分析：Parch, SibSp")
plt.subplot(211)
sns.countplot(x="Parch", hue="Survived", data=data_train)
plt.subplot(212)
sns.countplot(x="SibSp", hue="Survived", data=data_train)
plt.show()
data_train[["Parch", "Survived"]].groupby(["Parch"]).mean()
data_train[["SibSp", "Survived"]].groupby(["SibSp"]).mean()
for dataset in data_clean:
    dataset["Family"] = dataset["Parch"] + dataset["SibSp"]
sns.countplot(x="Family", hue="Survived", data=data_train)
data_train[["Family", "Survived"]].groupby(["Family"]).mean()
grid = sns.FacetGrid(data=data_train, row="Sex", col="Pclass", hue="Survived")
grid.map(sns.countplot, "Embarked")
grid.add_legend()
data_train[["Sex", "Pclass", "Embarked", "Survived"]].groupby(["Sex", "Pclass", "Embarked"]).mean()
data_train[["Sex", "Pclass", "Survived"]].groupby(["Sex", "Pclass"]).mean()
data_train[["Sex", "Embarked", "Survived"]].groupby(["Sex", "Embarked"]).mean()
pd.crosstab([data_train["Pclass"], data_train["Sex"]], data_train["Embarked"])
pd.crosstab(data_train["Pclass"], data_train["Embarked"])
data_train.head()
data_train.tail()
data_train.sample(5)
for dataset in data_clean:
    dataset["Title"] = dataset["Name"].str.extract(" ([A-Za-z]+)\.")
data_train["Title"].value_counts()
pd.crosstab(data_train["Title"], data_train["Sex"])
for dataset in data_clean:
    dataset["Title"] = dataset["Title"].replace(["Mme"], "Mrs")
    dataset["Title"] = dataset["Title"].replace(["Mlle", "Ms"], "Miss")
    dataset["Title"] = dataset["Title"].replace(["Lady", "Countess", "Capt",
                                                "Col", "Don", "Dr", "Major",
                                                "Rev", "Sir", "Jonkheer",
                                                "Dona"], "Rare")
data_train[["Title", "Survived"]].groupby(["Title"]).mean()
fig = plt.figure(figsize=(20, 8))
plt.subplot(211)
plt.title("不同年龄对存活率的影响")
sns.kdeplot(data=data_train.loc[data_train["Survived"]==0, "Age"], shade=True, label="not survived")
sns.kdeplot(data=data_train.loc[data_train["Survived"]==1, "Age"], shade=True, label="survived")
plt.subplot(212)
sns.distplot(data_train.loc[data_train["Survived"]==0, "Age"], bins=10, kde=False, label="not survived")
sns.distplot(data_train.loc[data_train["Survived"]==1, "Age"], bins=10, kde=False, label="survived")
plt.legend()
plt.show()
grid = sns.FacetGrid(data=data_train, row="Sex", col="Pclass", hue="Survived", xlim=[0,80])
grid.map(sns.kdeplot, "Age", shade=True)
grid.add_legend()
sns.boxplot(data=data_train, x="Pclass", y="Age")

for dataset in data_clean:
    dataset["AgeBand"] = pd.cut(dataset["Age"], bins=5)
data_train[["AgeBand", "Pclass", "Sex", "Survived"]].groupby(["AgeBand", "Pclass", "Sex"]).mean()

figure = plt.figure(figsize=(20, 8))
plt.title("不同Fare下的存活率")
plt.subplot(211)
sns.kdeplot(data_train.loc[data_train["Survived"]==0, "Fare"], shade=True, label="not survived")
sns.kdeplot(data_train.loc[data_train["Survived"]==1, "Fare"], shade=True, label="survived")
plt.subplot(212)
sns.distplot(data_train.loc[data_train["Survived"]==0, "Fare"], bins=20, label="not survived", kde=False)
sns.distplot(data_train.loc[data_train["Survived"]==1, "Fare"], bins=20, label="survived", kde=False)
plt.legend()
plt.show()

grid = sns.FacetGrid(data=data_train, col="Pclass", hue="Survived")
grid.map(sns.distplot, "Fare", kde=False, bins=10)
grid.add_legend()
for dataset in data_clean:
    dataset["FareBand"] = pd.qcut(data_train["Fare"], 4)
pd.crosstab(data_train["FareBand"], data_train["Pclass"])
data_train[["FareBand", "Survived"]].groupby(["FareBand"]).mean()
data_train.isnull().sum()
data_test.isnull().sum()
for dataset in data_clean:
    dataset["Embarked"].fillna(dataset["Embarked"].mode()[0], inplace=True)
data_train.isnull().sum()
for dataset in data_clean:
    dataset["Fare"].fillna(dataset["Fare"].median(), inplace=True)
    dataset["FareBand"] = pd.qcut(dataset["Fare"], 4)

data_test.isnull().sum()
for dataset in data_clean:
    dataset["Age"].fillna(dataset["Age"].median(), inplace=True)
    dataset["AgeBand"] = pd.cut(dataset["Age"], 5)
data_train.isnull().sum()
data_test.isnull().sum()
from sklearn.preprocessing import LabelEncoder
data_train.info()
label = LabelEncoder()
for dataset in data_clean:
    dataset["Sex_Code"] = label.fit_transform(dataset["Sex"])
    dataset["Embarked_Code"] = label.fit_transform(dataset["Embarked"])
    dataset["Title_Code"] = label.fit_transform(dataset["Title"])
    dataset["AgeBand_Code"] = label.fit_transform(dataset["AgeBand"])
    dataset["FareBand_Code"] = label.fit_transform(dataset["FareBand"])

data_train.info()

calc_columns = ["Pclass", "Family", "Sex_Code", "Embarked_Code", "Title_Code",
               "AgeBand_Code", "FareBand_Code"]

X_train = data_train[calc_columns]
y_train = data_train["Survived"]
X_test = data_test[calc_columns]
X_train.columns
sns.heatmap(X_train.corr(), annot=True)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, train_test_split, ShuffleSplit, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
abc = AdaBoostClassifier()
gbc = GradientBoostingClassifier()
lr = LogisticRegression()
p = Perceptron()
ls = LinearSVC()
s = SVC(probability=True)
gnb = GaussianNB()
mnb = MultinomialNB()
knc = KNeighborsClassifier()
mlpc = MLPClassifier()
xgb = XGBClassifier()
clf_list = [dtc, rfc, abc, gbc, lr, p, ls, s, gnb, mnb, knc, mlpc, xgb]

clf_compare = pd.DataFrame(columns=["name", "params", "mean_test_score", "test_score 3*sigma", "mean_train_score", "mean_fit_time"])
cv_splits = ShuffleSplit(n_splits=10, train_size=0.7, test_size=0.3, random_state=42)
row_index = 0
for clf in clf_list:
    clf_name = clf.__class__.__name__
    clf_compare.loc[row_index, "name"] = clf_name
    clf_compare.loc[row_index, "params"] = str(clf.get_params())
    cv_results = cross_validate(clf, X_train, y_train, cv=cv_splits, return_train_score=True)
    clf_compare.loc[row_index, "mean_test_score"] = np.mean(cv_results["test_score"])
    clf_compare.loc[row_index, "test_score 3*sigma"] = 3*np.std(cv_results["test_score"])
    clf_compare.loc[row_index, "mean_train_score"] = np.mean(cv_results["train_score"])
    clf_compare.loc[row_index, "mean_fit_time"] = np.mean(cv_results["fit_time"])
    row_index += 1;
clf_compare
s.fit(X_train, y_train)
y_test = s.predict(X_test)
test_data = pd.read_csv("./titanic/test.csv")
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": y_test
    })
submission.to_csv('./titanic/submission.csv', index=False)
param_grids = [
    {"criterion": ["gini", "entropy"], 
     "splitter": ["best", "random"],
    "max_depth": [3, 4, 5, 6, 7],
    "max_features": ["sqrt", "log2"]},  # DecisionTreeClassifier
    {"n_estimators": [50, 100, 150, 200, 250, 300],
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 7],
    "max_features": ["sqrt", "log2"]},  # RandomForest
    {"n_estimators": [50, 100, 150, 200, 250, 300],
    "learning_rate": [0.01, 0.03, 0.1, 0.3],
    "algorithm": ["SAMME", "SAMME.R"]},  # AdaBoost
    {"loss": ["deviance", "exponential"],
    "learning_rate": [0.01, 0.03, 0.1, 0.3],
    "n_estimators": [50, 100, 150, 200, 250, 300],
    "subsample": [0.1, 0.3, 0.5, 0.7, 1.0],
    "max_features": ["sqrt", "log2"]},  # GradientBoost
    {"penalty": ["l2"],
    "tol": [0.01, 0.001, 0.0001, 0.00001],
    "C": [0.3, 0.7, 1.0, 1.3, 1.7],
    "solver": ["lbfgs", "liblinear"],
    "max_iter": [100, 300, 500]},  # Logistic
    {"penalty": ["l2", "l1"],
    "alpha": [0.0001, 0.001, 0.01, 0.1, 1],
    "max_iter": [300,500, 700, 1000, 2000],
    "tol": [0.01, 0.001, 0.0001]},  # Perceptron
    {"penalty": ["l2"],
    "loss": ["hinge", "squared_hinge"],
    "tol": [0.01, 0.001, 0.0001],
    "C": [0.01, 0.03, 0.1, 0.3, 1, 3]},  # LinearSVC
    {"C": [0.01, 0.03, 0.1, 0.3, 1, 3],
    "kernel": ["rbf", "poly", "sigmoid"],
    "tol": [0.01, 0.001, 0.0001]},  # SVC
    {"var_smoothing": [1e-7, 1e-8, 1e-9]},  # GaussianNB
    {"alpha": [0, 0.3, 1.0, 1.3]},  # MultinomialNB
    {"n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "algorithm": ["ball_tree", "kd_tree"],
    "p": [1, 2]},  # KNN
    {"hidden_layer_sizes": [(50,), (100,), (150,), (200,)],
    "activation": ["identity", "logistic", "tanh", "relu"],
    "solver": ["lbfgs", "sgd", "adam"],
    "alpha": [0.0001, 0.001, 0.01, 0.1],
    "max_iter": [200, 300, 400, 500]},  # NeuralNetwork
    {"max_depth": [3, 5, 7, 9],
    "learning_rate": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    "booster": ["gbtree", "gblinear", "dart"]}  # XGBoost
]
gridsearch_compare = pd.DataFrame(columns=["name", "mean_train_score", "mean_test_score", "std_test_score*3", "best_params"])
row_index = 0
for clf, param_grid in zip(clf_list, param_grids):
    print(row_index)
    grid_clf = GridSearchCV(clf, param_grid, cv=cv_splits, n_jobs=-1, return_train_score=True)
    #grid_clf = RandomizedSearchCV(clf, param_grid, cv=cv_splits, n_jobs=-1, return_train_score=True, random_state=42, n_iter=30)
    grid_clf.fit(X_train, y_train)
    gridsearch_compare.loc[row_index, "name"] = clf.__class__.__name__
    cv_results = grid_clf.cv_results_
    gridsearch_compare.loc[row_index, "mean_train_score"] = np.mean(cv_results["mean_train_score"])
    gridsearch_compare.loc[row_index, "mean_test_score"] = np.mean(cv_results["mean_test_score"])
    gridsearch_compare.loc[row_index, "std_test_score*3"] = np.mean(cv_results["std_test_score"])*3
    gridsearch_compare.loc[row_index, "best_params"] = str(grid_clf.best_params_)
    row_index += 1
gridsearch_compare
voting_clf = [
    ("decisiontree", DecisionTreeClassifier(criterion="entropy", max_depth=7, max_features="log2", splitter="best")),
    ("randomforest", RandomForestClassifier(criterion="gini", max_depth=5, max_features="log2", n_estimators=50)),
    ("adaboost", AdaBoostClassifier(algorithm="SAMME.R", learning_rate=0.1, n_estimators=100)),
    ("gradientboost", GradientBoostingClassifier(learning_rate=0.01, loss="exponential", max_features="sqrt", n_estimators=250, subsample=1.0)),
    ("logistic", LogisticRegression(C=0.3, max_iter=100, penalty="l2", solver="lbfgs", tol=0.01)),
    ("linearsvc", LinearSVC(C=0.03, loss="hinge", penalty="l2", tol=0.01)),
    ("svc", SVC(C=0.1, kernel="poly", tol=0.01)),
    ("gaussian", GaussianNB(var_smoothing=1e-07)),
    ("multinomial", MultinomialNB(alpha=0)),
    ("knn", KNeighborsClassifier(algorithm="ball_tree", n_neighbors=7, p=1, weights="uniform")),
    ("mlp", MLPClassifier(activation="relu", alpha=0.01, hidden_layer_sizes=(50,), max_iter=300, solver="adam")),
    ("xgb", XGBClassifier(booster="gbtree", learning_rate=0.03, max_depth=3))
]
voting = VotingClassifier(voting_clf)

voting_cv_results = cross_validate(voting, X_train, y_train, cv=cv_splits, return_train_score=True)
voting_cv_results
np.mean(voting_cv_results["test_score"])
voting.fit(X_train, y_train)
y_test = voting.predict(X_test)
test_data = pd.read_csv("./titanic/test.csv")
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": y_test
    })
submission.to_csv('./titanic/submission.csv', index=False)
