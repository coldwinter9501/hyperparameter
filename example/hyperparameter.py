from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train_std, y_train)

y_pred = rf.predict(X_test_std)
print(accuracy_score(y_test, y_pred))

##grid search

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search
param_grid = {
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3,verbose = 2)
grid_search.fit(X_train_std, y_train)

best_grid = grid_search.best_estimator_
print(best_grid)
#grid_accuracy = evaluate(best_grid,X_test_std, y_test)

print(accuracy_score(best_grid.fit(X_train_std,y_train).predict(X_test_std)),y_test)


## random search
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# 使用随机网格搜索最佳超参数
# 首先创建要调优的基本模型
rf = RandomForestClassifier()
# 随机搜索参数，使用3倍交叉验证
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42)
# Fit模型
rf_random.fit(X_train_std, y_train)

print(rf_random.best_params_)


##hyperopt

def RFobjective(args):
    global X_train_std,y_train,y_test
    for parameter_name in ['n_estimators','max_depth']:
        args[parameter_name] = int(args[parameter_name])
    n_estimators = args['n_estimators']
    max_depth= args['max_depth']
    rf = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth)
    rf.fit(X_train_std, y_train)
    y_pred = rf.predict(X_test_std)
    return -accuracy_score(y_test, y_pred)

from hyperopt import fmin,tpe,hp

space = {
    'max_depth': hp.quniform('max_depth',5,10,1),
    'n_estimators': hp.quniform('n_estimators',100,200,2)#均匀分布
}
algo = tpe.suggest
best = fmin(RFobjective, space, algo= algo, max_evals=100)
print(best)

best_rf = RandomForestClassifier(max_depth=int(best['max_depth']),n_estimators=int(best['n_estimators']))
print(accuracy_score(best_rf.fit(X_train_std, y_train).predict(X_test_std), y_test))
