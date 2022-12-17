
from sklearn.metrics import f1_score, recall_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, plot_roc_curve, roc_auc_score, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
import pickle
from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from xgboost import XGBClassifier
from catboost import CatBoostClassifier


from sklearn.model_selection import GridSearchCV
warnings.filterwarnings('ignore')

from scipy.spatial.distance import cdist

#ATC stands for AUC Threshold Classification
#Threshold here is set to 0.7 for Recall0, so the function tries to maximize Recall1, while keeping the value of Recall0 higher than the threshold
#Function also depicts the ROC curve
def ATC(y_test, probas):
    Max_tprs = []
    if (type(y_test) is not list) & (type(probas) is not list):
        fpr, tpr, threshold = roc_curve(y_test, probas)
        max_tpr = 0
        min_fpr = 0
        best_threshold = 0
        for i in range(len(fpr)):
            if fpr[i] <= 0.3:
                if tpr[i] > max_tpr:
                    min_fpr = fpr[i]
                    max_tpr = tpr[i]
                    best_threshold = threshold[i]
        Max_tprs = max_tpr
    else:
        for t, p in zip(y_test, probas):
            fpr, tpr, threshold = roc_curve(t, p)
            max_tpr = 0
            min_fpr = 0
            best_threshold = 0
            for i in range(len(fpr)):
                if fpr[i] <= 0.3:
                    if tpr[i] > max_tpr:
                        min_fpr = fpr[i]
                        max_tpr = tpr[i]
                        best_threshold = threshold[i]
            Max_tprs += [max_tpr]

    plotAUC(y_test, probas, Max_tprs)

    return max_tpr

#Used only in the above function to depict the Roc curve
def plotAUC(truth, pred, Max_tprs, lab='model'):
    sns.set('talk', 'whitegrid', 'dark', font_scale=1, rc={"lines.linewidth": 2, 'grid.linestyle': '--'})
    if (type(truth) is not list) & (type(pred) is not list):
        fpr, tpr, _ = roc_curve(truth, pred)
        roc_auc = auc(fpr, tpr)
        lw = 2
        c = (np.random.rand(), np.random.rand(), np.random.rand())
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color=c, lw=lw, label=lab + '(AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.text(0.55, 0.18, 'max_tpr = ' + str(round(Max_tprs, 3)))
        plt.legend(loc="lower right")
    elif (type(truth) is list) & (type(pred) is list) & (len(pred) == len(truth)):
        shape = len(truth)
        plt.figure(figsize=(15, 5.5 * int(np.ceil(shape / 2))))
        for t, p, max_tpr, index in zip(truth, pred, Max_tprs, np.arange(shape)):
            fpr, tpr, _ = roc_curve(t, p)
            roc_auc = auc(fpr, tpr)
            lw = 2
            c = (np.random.rand(), np.random.rand(), np.random.rand())
            plt.subplot(int(np.ceil(shape / 2)), 2, index + 1)
            plt.plot(fpr, tpr, color=c, lw=lw, label=lab + '(AUC = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.text(0.55, 0.18, 'max_tpr = ' + str(round(max_tpr, 3)))
            plt.title('model_' + str(index), pad=10)
            plt.legend(loc="lower right")



#Output is the threshold, where 1 - Recall0 <= neg_thresh and Recall1 is maximized
#Used to get predictions instead of their probabilities
def best_threshold(y_test, probas, neg_thresh = 0.3):
    fpr, tpr, threshold = roc_curve(y_test, probas)

    max_tpr = 0
    min_fpr = 0
    best_threshold = 0
    for i in range(len(fpr)):
        if fpr[i] <= neg_thresh:
            if tpr[i] > max_tpr:
                min_fpr = fpr[i]
                max_tpr = tpr[i]
                best_threshold = threshold[i]
    return best_threshold




#Returns a dictionary with weights for positive and negative classes to balance the dataset
def class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    return class_weights

class MultEval:
    def __init__(self, model, x, y, neg_thresh=0.1):
        self.model = model
        self.x = x
        self.y = y
        self.neg_thresh = neg_thresh

    def aucs(self):
        auc_scores = {}
        for i, class_ in enumerate(np.unique(self.y)):
            y_ovr = np.where(self.y == i, 1, 0)
            auc_scores[class_] = roc_auc_score(y_ovr, self.model.predict_proba(self.x)[:, i])
        auc_scores['weighted'] = roc_auc_score(self.y, self.model.predict_proba(self.x)
                                               , average='weighted', multi_class='ovr')
        auc_scores['macro'] = roc_auc_score(self.y, self.model.predict_proba(self.x)
                                            , average='macro', multi_class='ovr')
        self.auc_scores = auc_scores

    def recalls(self):
        recall_scores = {}
        for i, class_ in enumerate(np.unique(self.y)):
            y_ovr = np.where(self.y == i, 1, 0)
            best_thresh = best_threshold(y_ovr, self.model.predict_proba(self.x)[:, i]
                                         , neg_thresh=self.neg_thresh)
            recall_scores[class_] = recall_score(y_ovr,
                                                 np.where(self.model.predict_proba(self.x)[:, i] <= best_thresh, 0, 1))

        self.recall_scores = recall_scores

    def accuracies(self):
        accuracy_scores = {}
        for i, class_ in enumerate(np.unique(self.y)):
            y_ovr = np.where(self.y == i, 1, 0)
            best_thresh = best_threshold(y_ovr, self.model.predict_proba(self.x)[:, i]
                                         , neg_thresh=self.neg_thresh)
            accuracy_scores[class_] = accuracy_score(y_ovr,
                                                     np.where(self.model.predict_proba(self.x)[:, i] <= best_thresh, 0,
                                                              1))

        self.accuracy_scores = accuracy_scores

    def evaluate(self):
        MultEval.aucs(self)
        MultEval.recalls(self)
        MultEval.accuracies(self)


def save_model(model, file):
    with open(file,'wb') as file:
        pickle.dump(model,file)

def read_model(file):
    with open(file,'rb') as file:
        model = pickle.load(file)
    return model

#Experimental function that returns multilabel predictions in terms of achieving the best f1-score
def get_multi_preds(model,x_test,y_test):
    predictions_base = pd.Series(np.repeat(1000,y_test.size), index = x_test.index)
    predictions = pd.DataFrame([])
    for i, class_ in enumerate(np.unique(y_test)):
        y_ovr = np.where(y_test == class_,1,0)
        pos_label = pd.Series(y_ovr).value_counts().index[1]
        best_thresh = best_f1_score(model,x_test,y_ovr, pos_label = pos_label, class_ = class_)
        preds = np.where(model.predict_proba(x_test)[:,i] < best_thresh,0,1)
        predictions[class_] = preds
    predictions.index = predictions_base.index
    predictions_base[(predictions.sum(axis = 1) == 1).values] = predictions.idxmax(axis = 1)
    new_predictions = predictions.copy()
    for i, class_ in enumerate(np.unique(y_test)):
        new_predictions[class_] = new_predictions[class_].replace(1,y_test[y_test == class_].size)
    predictions_base[(predictions.sum(axis = 1) != 1).values]  = new_predictions.idxmax(axis = 1)
    return predictions_base