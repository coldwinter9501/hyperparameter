'''
yperopt库为python中的模型选择和参数优化提供了算法和并行方案。
机器学习常见的模型有KNN,SVM，PCA，决策树，GBDT等一系列的算法，
但是在实际应用中，我们需要选取合适的模型，并对模型调参，
得到一组合适的参数。尤其是在模型的调参阶段，需要花费大量的时间和精力，
却又效率低下。但是我们可以换一个角度来看待这个问题，模型的选取，
以及模型中需要调节的参数，可以看做是一组变量，模型的质量标准（比如正确率，AUC）
等等可以看做是目标函数，这个问题就是超参数的优化的问题。我们可以使用搜索算法来解决。
Hyperopt提供了一个优化接口，这个接口接受一个评估函数和参数空间，
能计算出参数空间内的一个点的损失函数值。用户还要指定空间内参数的分布情况。
Hyheropt四个重要的因素：指定需要最小化的函数，搜索的空间，
采样的数据集(trails database)（可选），搜索的算法（可选）。
'''
#coding:utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from random import shuffle
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
import pickle
import time
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK

def loadFile(fileName = "./browsetop200Pca.csv"):
    data = pd.read_csv(fileName,header=None)
    data = data.values
    return data

data = loadFile()
label = data[:,-1]
attrs = data[:,:-1]
labels = label.reshape((1,-1))
label = labels.tolist()[0]

minmaxscaler = MinMaxScaler()
attrs = minmaxscaler.fit_transform(attrs)

index = range(0,len(label))
shuffle(index)
trainIndex = index[:int(len(label)*0.7)]
print(len(trainIndex))
testIndex = index[int(len(label)*0.7):]
print(len(testIndex))
attr_train = attrs[trainIndex,:]
print(attr_train.shape)
attr_test = attrs[testIndex,:]
print(attr_test.shape)
label_train = labels[:,trainIndex].tolist()[0]
print(len(label_train))
label_test = labels[:,testIndex].tolist()[0]
print(len(label_test))
print(np.mat(label_train).reshape((-1,1)).shape)


def GBM(argsDict):
    max_depth = argsDict["max_depth"] + 5
    n_estimators = argsDict['n_estimators'] * 5 + 50
    learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
    subsample = argsDict["subsample"] * 0.1 + 0.7
    min_child_weight = argsDict["min_child_weight"]+1
    print( "max_depth:" + str(max_depth))
    print ("n_estimator:" + str(n_estimators))
    print ("learning_rate:" + str(learning_rate))
    print ("subsample:" + str(subsample))
    print ("min_child_weight:" + str(min_child_weight))
    global attr_train, label_train

    gbm = xgb.XGBClassifier(nthread=4,    #进程数
                            max_depth=max_depth,  #最大深度
                            n_estimators=n_estimators,   #树的数量
                            learning_rate=learning_rate, #学习率
                            subsample=subsample,      #采样数
                            min_child_weight=min_child_weight,   #孩子数
                            max_delta_step = 10,  #10步不降则停止
                            objective="binary:logistic")

    metric = cross_val_score(gbm, attr_train,label_train, cv=5, scoring="roc_auc").mean()
    print(metric)
    return -metric

space = {"max_depth":hp.randint("max_depth",15),
         "n_estimators":hp.randint("n_estimators",10),  #[0,1,2,3,4,5] -> [50,]
         "learning_rate":hp.randint("learning_rate",6),  #[0,1,2,3,4,5] -> 0.05,0.06
         "subsample":hp.randint("subsample",4),#[0,1,2,3] -> [0.7,0.8,0.9,1.0]
         "min_child_weight":hp.randint("min_child_weight",5), #
        }
algo = partial(tpe.suggest,n_startup_jobs=1)
best = fmin(GBM, space, algo=algo, max_evals=4)

print(best)
print(GBM(best))
