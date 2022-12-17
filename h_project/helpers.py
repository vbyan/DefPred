
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import  roc_auc_score
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import warnings
from dateutil import parser
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import pickle

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


#Weighted objective function that penalizes either the misclassified ones or misclassified zeros
#beta is the ratio (if beta is 2, then the penalization for misclassified posititve class is two times higher)
def weighted_logloss(beta, scale_pos_weight):
    def custom_loss(y_train,y_hat):
        y = y_train
        weights = np.where(y == 1.0, scale_pos_weight, 1.0)
        p = 1. / (1. + np.exp(-y_hat))
        grad = p * (beta + y - beta*y) - y
        hess = p * (1 - p) * (beta + y - beta*y)
        return weights * grad, weights * hess
    return custom_loss

#This can passed as a scorer in GridSearchCV, cross_val_score and some other methods
#The score is the maximal values of Recall1, when Recall0 >= 0.7
def custom_scorer(y_test, probas):
    fpr, tpr, threshold = roc_curve(y_test, probas)
    auc = roc_auc_score(y_test,probas)

    max_tpr = 0
    min_fpr = 0
    best_threshold = 0
    for i in range(len(fpr)):
        if fpr[i] <= 0.3:
            if tpr[i] > max_tpr:
                min_fpr = fpr[i]
                max_tpr = tpr[i]
                best_threshold = threshold[i]

    return  (auc + 2*max_tpr)/3



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



#Provides some statistics about the profitability of the model based on only the Contract Amount
class Revenue:
    def __init__(self, model, x_test, y_test, revenue_col, thresh = None):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.revenue_col = revenue_col
        if thresh == 'best':
            best_thresh = best_threshold(y_test, model.predict_proba(x_test)[:, 1])
        elif thresh is None:
            best_thresh = 0.5
        else:
            best_thresh = best_thresh = best_threshold(y_test, model.predict_proba(x_test)[:, 1], neg_thresh = thresh)
        if type(x_test) is list:
            for i in range(len(x_test)):
                x_test[i].index = np.arange(y_test.shape[0])
        else:
            x_test.index = np.arange(y_test.shape[0])
        validation = pd.DataFrame([])
        validation['pred'] = np.where(model.predict_proba(x_test)[:, 1] <= best_thresh, 0, 1)
        validation['true'] = y_test
        validation['Limit'] = revenue_col / 1000000
        validation['pred_result'] = np.where(validation['true'] > validation['pred'],
                                             'False negative', np.where(validation['true'] < validation['pred'],
                                                                        'False positive',
                                                                        np.where(validation['true'] == 1,
                                                                                 'True positive', 'True negative')))
        self.validation = validation

    @property
    def conf_mean_(self):
        return self.validation.groupby('pred_result')['Limit'].mean().apply(lambda x: round(x, 2))

    @property
    def conf_sum_(self):
        return self.validation.groupby('pred_result')['Limit'].sum().apply(lambda x: round(x, 2))

    @property
    def total_loss_(self):
        return round(
            self.validation[self.validation['pred_result'].isin(['False positive', 'False negative'])]['Limit'].sum(),
            2)

    @property
    def total_save_(self):
        return round(
            self.validation[self.validation['pred_result'].isin(['True positive', 'True negative'])]['Limit'].sum(), 2)



    @property
    def report_(self):
        print('    SUM (million dram)')
        print()
        print(str(self.conf_sum_)[12:-27])
        print('--------------------------------')
        print('    MEAN (million dram)')
        print()
        print(str(self.conf_mean_)[12:-27])
        print('--------------------------------')
        print('Total Loss (million dram)')
        print()
        print(self.total_loss_)
        print('--------------------------------')
        print('Total Save (million dram)')
        print()
        print(self.total_save_)
        print('--------------------------------')
        print('Total Loss prct')
        print()
        print(str(round((self.total_loss_ / (self.total_save_ + self.total_loss_)) * 100, 1)) + ' %')
        print('--------------------------------')


#While oversampling huge databases, somtemies MemoryError may occur
#It's recommended to use this function instead to oversample the dataset for parts
def batch_oversampling(oversampler, x,y, k_neighbors_perc = 50, batch_size = 1000):
    entire_oversampled_x = pd.DataFrame(np.zeros((1,x.shape[1])), columns = x.columns)
    entire_oversampled_y = np.array([0])
    for i in range(batch_size, x.shape[0], batch_size):
        pos_values_count = y[i - batch_size:i].sum()
        k_neighbors = int(pos_values_count * k_neighbors_perc / 100)
        oversampler.k_neighbors = k_neighbors
        a,b = oversampler.fit_resample(x.iloc[i - batch_size:i], y[i-batch_size:i])
        a = pd.DataFrame(a,columns = x.columns)
        entire_oversampled_x = pd.concat([entire_oversampled_x,a], axis = 0, ignore_index = True)
        entire_oversampled_y = np.hstack((entire_oversampled_y.reshape(-1), b.reshape(-1)))
    pos_values_count = y[i:x.shape[0]].sum()
    k_neighbors = int(pos_values_count * k_neighbors_perc / 100)
    oversampler.k_neighbors = k_neighbors
    residual_x, residual_y = oversampler.fit_resample(x.iloc[i:x.shape[0]],y[i:x.shape[0]])
    entire_oversampled_x = pd.concat([entire_oversampled_x, residual_x], axis = 0, ignore_index = True).iloc[1:,:]
    entire_oversampled_y = np.hstack((entire_oversampled_y.reshape(-1), residual_y.reshape(-1)))[1:]
    return entire_oversampled_x, entire_oversampled_y

def parser_with_nan(x):
    try:
        return parser.parse(x)
    except:
        return np.nan


#This is merging loan-data with behaviroal databases such as CRRC and WVS
#Algorithm implements initialization using overlapping features in both dataset
#Initialization is performed by minimizing the difference (distance) between the corresponding values of overlapping features,
# so we find the person in behavioral dataset that is the same as in loan dataset or is slightly different
class merge_overlaps:
    def __init__(self, data1, data2, overlaps1, overlaps2, weights=None, warm_start = False,directory = None):
        self.data1 = data1
        self.data2 = data2
        self.overlaps1 = overlaps1
        self.overlaps2 = overlaps2
        self.weights = weights
        self.warm_start = warm_start
        self.directory = directory

    #distances between categorical features
    def cat_dist(x, y):
        return cdist(x, y, metric='hamming')

    # distances between numerical features
    def num_dist(x, y):
        return cdist(x, y, metric='euclidean')

    def compute_distances(self,X, Y, weights=None):
        categoricals = locate_categoricals(X)
        numericals = list(set(np.arange(X.shape[1]).tolist()) - set(categoricals))

        num_weights = list(map(weights.__getitem__, numericals))
        cat_weights = list(map(weights.__getitem__, categoricals))

        if self.warm_start:
            with open(self.directory + 'minmaxscalers_x.txt','rb') as file:
                minmaxscalers_x = pickle.load(file)
            with open(self.directory + 'minmaxscalers_y.txt','rb') as file:
                minmaxscalers_y = pickle.load(file)
            with open(self.directory + 'minmaxscaler_numdist.txt','rb') as file:
                minmaxscaler_numdist = pickle.load(file)

        else:
            minmaxscalers_x = {}
            minmaxscalers_y = {}
            minmaxscaler_numdist = MinMaxScaler()
        for col in numericals:
            if self.warm_start:
                X.iloc[:, col] = minmaxscalers_x[col].transform(X.iloc[:, col].values.reshape(-1, 1))
                Y.iloc[:, col] = minmaxscalers_y[col].transform(Y.iloc[:, col].values.reshape(-1, 1))
            else:
                scaler_x = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X.iloc[:, col] = scaler_x.fit_transform(X.iloc[:, col].values.reshape(-1, 1))
                Y.iloc[:, col] = scaler_y.fit_transform(Y.iloc[:, col].values.reshape(-1, 1))
                minmaxscalers_x[col] = scaler_x
                minmaxscalers_y[col] = scaler_y


        cat_dist_matrix = pd.DataFrame(np.zeros((X.shape[0], Y.shape[0])))
        num_dist_matrix = pd.DataFrame(np.zeros((X.shape[0], Y.shape[0])))
        for cat_feature_index, w in zip(categoricals, cat_weights):
            cat_dist_matrix += w * merge_overlaps.cat_dist(X.iloc[:, cat_feature_index].values.reshape(-1, 1),
                                                           Y.iloc[:, cat_feature_index].values.reshape(-1, 1)) / sum(
                cat_weights)
        for num_feature_index, w in zip(numericals, num_weights):
            num_dist_matrix += w * merge_overlaps.num_dist(X.iloc[:, num_feature_index].values.reshape(-1, 1),
                                                           Y.iloc[:, num_feature_index].values.reshape(-1, 1)) / sum(
                num_weights)

        #Categorical and numerical distance matrices are added up together using some weights,
        #because the variance in numerical features is always lower than in categoricals
        if self.warm_start:
            num_dist_matrix = minmaxscaler_numdist.transform(num_dist_matrix)
        else:
            num_dist_matrix = minmaxscaler_numdist.fit_transform(num_dist_matrix)
        entire_dist_matrix = (cat_dist_matrix + 2 * num_dist_matrix) / 3

        if not self.warm_start:
            if self.directory:
                with open(self.directory + 'minmaxscalers_x.txt', 'wb') as file:
                    pickle.dump(minmaxscalers_x,file)
                with open(self.directory + 'minmaxscalers_y.txt', 'wb') as file:
                    pickle.dump(minmaxscalers_y,file)
                with open(self.directory + 'minmaxscaler_numdist.txt', 'wb') as file:
                    pickle.dump(minmaxscaler_numdist,file)

        return pd.DataFrame(entire_dist_matrix)

    def find_matches(self):
        distance_matrix = merge_overlaps.compute_distances(self,self.overlaps1, self.overlaps2, weights=self.weights)
        self.distance_matrix = distance_matrix
        matching_indices = distance_matrix.idxmin(axis=1).values.tolist()
        matching_values = distance_matrix.min(axis=1).values.tolist()
        self.matching_indices = matching_indices
        self.matching_values = matching_values
        return matching_indices

    def merge(self):
        merge_overlaps.find_matches(self)
        left_merge = self.data2.iloc[self.matching_indices]
        left_merge.index = self.overlaps1.index
        right_merge = self.data1.loc[left_merge.index.values.tolist()]
        merged_data = pd.concat([right_merge, left_merge], axis=1)
        return merged_data

    @property
    def distance_matrix_(self):
        return self.distance_matrix

    @property
    def matching_(self):
        return pd.concat([pd.Series(self.matching_indices), pd.Series(self.matching_values)], axis=1)

#Returns a dictionary with weights for positive and negative classes to balance the dataset
def class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    return class_weights

#Finds categorical features by its format. Integers are considered as categoricals, other types as numericals.
def locate_categoricals(data):
    categorical_indices = []
    categorical_columns  = data.select_dtypes(['int','int64','object','int32']).columns.values
    for column in categorical_columns:
        categorical_indices += [np.where(data.columns.values == column)[0][0]]
    return categorical_indices
