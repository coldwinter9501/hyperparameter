import pandas as pd
import numpy as np
import glob
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score, roc_curve, auc

def load_data(file_path):
    grainwise_flist = glob.glob(file_path)
    grainwise_data = pd.DataFrame()
    dataset_keys = ['001_IPF_0', '001_IPF_1', '001_IPF_2', '100_IPF_0', '100_IPF_1', '100_IPF_2', '111_IPF_0',
                    '111_IPF_1', '111_IPF_2','AspectRatios_0', 'AspectRatios_1', 'AvgMisorientations',
                    'EquivalentDiameters','F1List', 'F1sptList', 'F7List', 'FeatureBoundaryElementFrac',
                    'FeatureVolumes', 'GBEuc','KernelAvg','Neighborhoods', 'NumCells', 'NumNeighbors',
                    'Omega3s', 'QPEuc', 'Schmid', 'SharedSurfaceAreaList','SurfaceAreaVolumeRatio',
                    'SurfaceFeatures', 'TJEuc', 'hotspot', 'mPrimeList', 'Euler_1', 'Euler_2', 'Euler_3',
                    'Taylor']
    for i,f in enumerate(grainwise_flist):
        temp = pd.read_csv(f)
        temp['file_source'] = i+1
        grainwise_data = grainwise_data.append(temp[dataset_keys])
    grainwise_data.loc[grainwise_data['hotspot'] == False, 'hotspot'] = 0
    grainwise_data.loc[grainwise_data['hotspot'] == True, 'hotspot'] = 1
    return grainwise_data

def plotROC(tpr, fpr, label=''):
    plt.plot(fpr, tpr, label=label)
    plt.legend()
    plt.ylabel('True positive rate.')
    plt.xlabel('False positive rate')
    plt.savefig('ROC')
    plt.show()

def assessMod(predsTrain, yTrain, predsValid=[], yValid=[],report=True, plot=True):
    trainAcc = accuracy_score(yTrain, np.round(predsTrain))
    fprTrain, tprTrain, thresholdsTrain = roc_curve(yTrain, predsTrain)
    trainAUC = auc(fprTrain, tprTrain)

    if predsValid != []:
        accuracy_score(yValid, np.round(predsValid))
        fprValid, tprValid, thresholdsValid = roc_curve(yValid, predsValid)
        validAcc = accuracy_score(yValid, np.round(predsValid))
        validAUC = auc(fprValid, tprValid)
    else:
        validAcc = np.nan
        fprValid = np.nan
        tprValid = np.nan
        validAUC = np.nan
    if report:
        print('Train accuracy:', trainAcc, '| Train AUC:',
              trainAUC)
        if not isinstance(predsValid, list):
            print('Validation accuracy:', validAcc, '| Test AUC:',
                  validAUC)
        print('-' * 30)
    # Plot
    if plot:
        plotROC(tprTrain, fprTrain, label='Train')
        if not isinstance(predsValid, list):
            plotROC(tprValid, fprValid, label='Valid')
    # Stats output
    stats = {'fprTrain': fprTrain,
             'fprValid': fprValid,
             'tprTrain': tprTrain,
             'tprValid': tprValid,
             'trainAcc': trainAcc,
             'validAcc': validAcc,
             'trainAUC': trainAUC,
             'validAUC': validAUC}
    return stats