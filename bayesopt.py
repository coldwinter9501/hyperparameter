import pandas as pd
import numpy as np
import glob
import lightgbm as lgb
from utils import *


path = 'G:/Hotspot/FCC_grainwise/*'
grainwise_data = load_data(path)
hotspot_count = grainwise_data['hotspot'].value_counts()
print(hotspot_count)
y = grainwise_data.pop('hotspot')
dataset = grainwise_data.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.4,random_state=7,stratify=y)
print('Train shape: ', X_train.shape)
print('Test shape: ', X_test.shape)

####SMOTE
from collections import Counter
from imblearn.over_sampling import SMOTE
imbalanceHandeler =  SMOTE(random_state=2019, ratio='auto', kind='regular', k_neighbors=5, n_jobs=-1)
X_resampled_training, y_resampled_training = imbalanceHandeler.fit_sample(X_train, y_train)
print("Count of Samples Per Class in balanced state:")
print(sorted(Counter(y_resampled_training).items()))
dataset_data_training = X_resampled_training
dataset_target_training = y_resampled_training

print(np.sum(dataset_target_training == 0))
print(np.sum(dataset_target_training == 1))

#we will use the common classification metric of Receiver Operating
# Characteristic Area Under the Curve (ROC AUC).
# Model with default hyperparameters
model = lgb.LGBMClassifier()
print(model)
from sklearn.metrics import roc_auc_score
model.fit(dataset_data_training, dataset_target_training, eval_set=[(X_test, y_test)], early_stopping_rounds=5)
predictions = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, predictions)
#print('The baseline score on the test set is {:.4f}.'.format(auc))


##构建lgb数据集
trainData = lgb.Dataset(dataset_data_training, label=dataset_target_training, free_raw_data=False,
                            feature_name=list(grainwise_data.columns),
                            categorical_feature= 'auto')

testData = lgb.Dataset(X_test, label=y_test, free_raw_data=False,
                            feature_name=list(grainwise_data.columns),
                            categorical_feature= 'auto')

import csv
from hyperopt import STATUS_OK

def objective(params, n_folds=10):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
    global ITERATION
    ITERATION += 1
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'min_child_samples', 'n_estimators']:
        params[parameter_name] = int(params[parameter_name])
    # Perform n_folds cross validation
    cv_results = lgb.cv(params, trainData, num_boost_round=10000, nfold=n_folds,
                        early_stopping_rounds=100, metrics='auc', seed=50)
    # Extract the best score
    best_score = np.max(cv_results['auc-mean'])
    # Loss must be minimized
    loss = 1 - best_score
    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)
    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators])
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators,
            'status': STATUS_OK}

from hyperopt import hp
from hyperopt.pyll.stochastic import sample
# Define the search space
space = {
    'num_leaves': hp.quniform('num_leaves', 30, 60, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),#[exp(low),exp(high)]
    'min_child_samples': hp.quniform('min_child_samples', 20, 200, 5),#均匀分布
    'n_estimators': hp.quniform('num_estimators',100,200,2)#均匀分布
}

# Sample from the full space
x = sample(space)

from hyperopt import tpe
from hyperopt import Trials

# Keep track of results
bayes_trials = Trials()
# File to save first results
out_file = './gbm_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators'])
of_connection.close()

from hyperopt import fmin
# Global variable
global ITERATION
ITERATION = 0
MAX_EVALS = 10
# Run optimization
best = fmin(fn= objective, space= space, algo = tpe.suggest,
            max_evals = MAX_EVALS, trials= bayes_trials, rstate = np.random.RandomState(50))
print(best)
# Sort the trials with lowest loss (highest AUC) first
bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
#print(bayes_trials_results[:2])

results = pd.read_csv('./gbm_trials.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending= True, inplace = True)
results.reset_index(inplace = True, drop = True)
results.head()
#print(results['params'][0])


for param in best.keys():
    if param in ['min_child_samples','num_estimators','num_leaves']:
        best[param] = int(best[param])

# Train
gbm = lgb.train(best,
                trainData,
                1000,
                valid_sets=[trainData, testData],
                verbose_eval=4)

# Plot importance
#lgb.plot_importance(gbm)
#plt.show()

feature = grainwise_data.columns.tolist()[:]
feat_imp = pd.Series(gbm.feature_importance("split"), index=feature).sort_values(ascending=False)
fig,ax = plt.subplots(figsize=(30,10))
plt.tick_params(labelsize=23)
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }

feat_imp.plot(kind='bar', ax = ax)
plt.title('Feature Importances for split',font2)
plt.ylabel('Feature Importance Score',font2)
plt.savefig('Feature Importances for split.png')
plt.show()

feat_imp = pd.Series(gbm.feature_importance("gain"), index=feature).sort_values(ascending=False)
fig,ax = plt.subplots(figsize=(30,10))
plt.tick_params(labelsize=23)
feat_imp.plot(kind='bar', ax = ax)
plt.title('Feature Importances for gain',font2)
plt.ylabel('Feature Importance Score',font2)
plt.savefig('Feature Importances for gain.png')
plt.show()

print('The baseline score on the test set is {:.4f}.'.format(auc))

# Predict
predsTrain1 = gbm.predict(X_train, num_iteration=gbm.best_iteration)
predsTest1 = gbm.predict(X_test, num_iteration=gbm.best_iteration)

assessMod(predsTrain1, y_train,
          predsValid = predsTest1, yValid = y_test,
          report=True, plot=True)
