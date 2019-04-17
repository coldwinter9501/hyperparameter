import pandas as pd
import numpy as np
import glob
import lightgbm as lgb
from utils import *

path = 'G:/Hotspot/FCC_grainwise/*'
grainwise_data = load_data(path)
print(grainwise_data.shape)
hotspot_count = grainwise_data['hotspot'].value_counts()
print(hotspot_count)
y = grainwise_data.pop('hotspot')
dataset = grainwise_data.values

import matplotlib.pyplot as plt
plt.hist(y, edgecolor = 'k')
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Counts of Labels')
plt.savefig('Counts of labels.png')
plt.show()

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

##构建lgb数据集
trainData = lgb.Dataset(dataset_data_training, label=dataset_target_training, free_raw_data=False,
                            feature_name=list(grainwise_data.columns),
                            categorical_feature = 'auto')

testData = lgb.Dataset(X_test, label=y_test, free_raw_data=False,
                            feature_name=list(grainwise_data.columns),
                            categorical_feature = 'auto')

params = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'binary',
          'nthread': 5,
          'num_leaves': 64,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'binary_error'}
# Train model
gbm = lgb.train(params,
                trainData,
                1000, # Max number of trees
                valid_sets=[trainData, testData],
                early_stopping_rounds = 50,
                verbose_eval=4)

# Run prediction on training, validation, and test sets using raw data
# Specify which model iteration to use
predsTrain = gbm.predict(X_train, num_iteration=gbm.best_iteration)
predsTest = gbm.predict(X_test, num_iteration=gbm.best_iteration)

from sklearn.metrics import accuracy_score, roc_curve, auc

# Report model performance on training and validation sets
assessMod(predsTrain, y_train,
          predsValid=predsTest, yValid=y_test,
          report=True, plot=True)

# Plot importance
#lgb.plot_importance(gbm, importance_type="split", title="split")
#plt.show()
#lgb.plot_importance(gbm, importance_type="gain", title='gain')
#plt.show()

# Importance values are also available in:
#print(gbm.feature_importance("split"))
#print(gbm.feature_importance("gain"))


feature = grainwise_data.columns.tolist()[:]
feat_imp = pd.Series(gbm.feature_importance("split"), index=feature).sort_values(ascending=False)
fig,ax = plt.subplots(figsize=(30,10))
#fig.set_size_inches(30,10)
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
