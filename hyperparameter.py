import pandas as pd
import numpy as np
import glob
import lightgbm as lgb
from utils import *


path =  'F:/Hotspot/FCC_grainwise/*'
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
model

from sklearn.metrics import roc_auc_score

model.fit(dataset_data_training, dataset_target_training, eval_set=[(X_test, y_test)], early_stopping_rounds=5)
predictions = model.predict_proba(X_test)[:, 1]
print(predictions)
auc = roc_auc_score(y_test, predictions)
print('The baseline score on the test set is {:.4f}.'.format(auc))

#Create parameters to search
from sklearn.model_selection import GridSearchCV
gridParams = {
    'learning_rate': [0.01,0.05,0.1],
    'n_estimators': [24,32,64],
    'num_leaves': [24,36,48],
    'subsample' : [0.7,1]
    }
# Model with default hyperparameters
model = lgb.LGBMClassifier()
# To view the default model params:
model.get_params().keys()
# Create the grid
grid = GridSearchCV(model, gridParams, verbose=1,scoring="roc_auc", cv=5, n_jobs=-1)
# Run the grid
grid.fit(X_train, y_train)
# Print the best parameters found
print(grid.best_params_)
print(grid.best_score_)

# Using parameters already set above, replace in the best from the grid search
params = {}
params['learning_rate'] = grid.best_params_['learning_rate']
params['num_leaves'] = grid.best_params_['num_leaves']
params['subsample'] = grid.best_params_['subsample']
params['n_estimators'] = grid.best_params_['n_estimators']

gbm = lgb.LGBMClassifier(params)
gbm.fit(dataset_data_training, dataset_target_training, eval_set=[(X_test, y_test)], early_stopping_rounds=5)
predictions = gbm.predict_proba(X_test)[:, 1]
print(predictions)
auc = roc_auc_score(y_test, predictions)
print('The best gridsearch model roc is {:.4f}.'.format(auc))
# Train
#gbm = lgb.train(params,
 #               trainData,
  #              100000,
   #             valid_sets=[trainData, testData],
    #            early_stopping_rounds = 50,
     #           verbose_eval=4)


# Plot importance
#lgb.plot_importance(gbm)
#plt.show()

# Predict
predsTrain1 = gbm.predict(X_train, num_iteration=gbm.best_iteration)
predsTest1 = gbm.predict(X_test, num_iteration=gbm.best_iteration)

assessMod(predsTrain1, y_train,
          predsValid = predsTest1, yValid = y_test,
          report=True, plot=True)