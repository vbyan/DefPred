import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from sklearn.base import clone
from preprocessing import get_data

#Computes all loss or profit-related variables
#Those variables may be used as attributes of this object
#Also provides a report_ property attribute which reports all the necessary loss_related info

class Revenue:
    def __init__(self, x_test, y_test, y_pred, y_defaulttime):
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.defaulttime = y_defaulttime

    def get_true_pred_df(self):
        tp_df = pd.DataFrame([])
        tp_df['true'] = self.y_test
        tp_df['pred'] = self.y_pred
        tp_df.index = self.x_test.index
        self.tp_df = tp_df

    def get_cases(self):
        false_positives = self.tp_df[(self.tp_df['pred'] == 1) & (self.tp_df['true'] == 0)].index.values
        false_negatives = self.tp_df[(self.tp_df['pred'] == 0) & (self.tp_df['true'] == 1)].index.values
        defaults = self.tp_df[self.tp_df['true'] == 1].index.values

        self.fp = false_positives
        self.fn = false_negatives
        self.defaults = defaults

    def get_parameters(self):
        mortgage = self.x_test['FMORTGAGESUM']
        guarantee = self.x_test['FGUARSUM']
        amount = self.x_test['FAGRSUM']
        percent = self.x_test['FPCNDER'].apply(lambda x: x / 100)
        n_months = self.x_test['FDAYQUAN'].apply(lambda x: int(round(x / 30, 0)))


        self.mortgage = mortgage
        self.guarantee = guarantee
        self.amount = amount
        self.percent = percent
        self.n_months = n_months

    def annuity_coef(percent, n_periods):
        percent = percent / 12
        return n_periods * (percent * (1 + percent) ** n_periods) / \
               ((1 + percent) ** n_periods - 1) - 1

    def compute_loss(self):
        initial_loss = self.amount.loc[self.defaults] * (1-self.defaulttime.loc[self.defaults]) - self.mortgage.loc[self.defaults] - \
                       self.guarantee.loc[self.defaults]
        fp_loss = self.amount.loc[self.fp] * \
                  Revenue.annuity_coef(self.percent.loc[self.fp], self.n_months.loc[self.fp])
        fn_loss = self.amount.loc[self.fn]* (1-self.defaulttime.loc[self.defaults]) - self.mortgage.loc[self.fn] - \
                  self.guarantee.loc[self.fn]

        self.initial_loss = int('{:.0f}'.format(initial_loss.sum()))
        self.fp_loss = int('{:.0f}'.format(fp_loss.sum()))
        self.fn_loss = int('{:.0f}'.format(fn_loss.sum()))
        self.final_loss = int('{:.0f}'.format(fp_loss.sum() + fn_loss.sum()))

        reduced_loss_prct = int('{:.0f}'.format(100 * (1 - self.final_loss / self.initial_loss)))
        reduced_loss = int('{:.0f}'.format(self.initial_loss - self.final_loss))
        reduced_loss_mean = int('{:.0f}'.format(reduced_loss / self.x_test.shape[0]))

        self.reduced_loss_prct = reduced_loss_prct
        self.reduced_loss = reduced_loss
        self.reduced_loss_mean = reduced_loss_mean

    def get_statistics(self):
        Revenue.get_true_pred_df(self)
        Revenue.get_cases(self)
        Revenue.get_parameters(self)
        Revenue.compute_loss(self)

    @property
    def report_(self):
        if hasattr(Revenue,'initial_loss'):
            pass
        else:
             Revenue.get_statistics(self)
        print('Initial loss: ' + str(self.initial_loss))
        print('False Positive Loss: ' + str(self.fp_loss))
        print('False Negative Loss: ' + str(self.fn_loss))
        print('Final loss: ' + str(self.final_loss))
        print('-------------------------------')
        print('Initial loss is reduced by: ' + str(self.reduced_loss_prct) + '%')
        print('Initial loss is reduced by: ' + str(self.reduced_loss) + ' AMD')
        print('Initial loss in average is reduced by: ' + str(self.reduced_loss_mean) + ' AMD')

#Finds best threshold by computing the loss for every threshold in the given list
#Uses the Revenue object to compute the loss
class BestThresh:
    def __init__(self, x_test, y_test,probas,y_defaulttime, n_threshes=100):

        self.x_test = x_test
        self.y_test = y_test
        self.n_threshes = n_threshes
        self.probas = probas
        self.defaulttime = y_defaulttime

    def get_df(self):
        d = pd.DataFrame([])
        d['threshold'] = np.linspace(1e-5, 1, self.n_threshes)
        d['revenue'] = d['threshold'].apply(lambda x: Revenue(self.x_test,
                                                              self.y_test,
                                                              np.where(self.probas < x,
                                                                       0, 1),self.defaulttime))
        d['revenue'].apply(lambda x: x.get_statistics())
        d['reduced_loss'] = d['revenue'].apply(lambda x: x.reduced_loss)
        d['reduced_loss_prct'] = d['revenue'].apply(lambda x: x.reduced_loss_prct)

        return d

    def get_best_thresh(self):
        d = BestThresh.get_df(self)
        max_index = d['reduced_loss'].idxmax()
        best_thresh = d['threshold'].loc[max_index]

        self.best_thresh = best_thresh
        self.d = d
        self.threshes = np.linspace(1e-5, 1, self.n_threshes)
        self.percents = d['reduced_loss_prct']

    def plot(self):
        def plot_(self):
            plt.figure(figsize=(10, 5))
            plt.plot(self.threshes, self.percents, c='g')
            plt.ylim(0)
            plt.xlabel('Thresholds', fontsize=16, fontweight='bold')
            plt.ylabel('Loss reduced by %', fontsize=16, fontweight='bold')

        if hasattr(BestThresh, 'best_thresh'):
            plot_(self)
        else:
            BestThresh.get_best_thresh(self)
            plot_(self)



#Is used to automatically find the best threshold without overfitting its values on test set
#In order to do that the test set is not included in the process at all
#So the best threshold is being found by the way of cross-validation on the train set
#Best threshold for every seperation is acquired using the BestThresh object
class ThreshValidation:
    def __init__(self, model, x, y,y_defaulttime, cv=5, n_jobs=-1,loan_type = None):
        self.model = clone(model)
        self.x = x
        self.y = y
        self.cv = cv
        self.n_jobs = n_jobs
        self.loan_type = loan_type
        self.defaulttime = y_defaulttime

    def get_loan_types(self):
        type_x,type_y = get_data(self.x,self.y,loan_type = self.loan_type,train_test = 'train')
        loan_type_indices = type_x.index.values

        self.loan_type_indices = loan_type_indices

    def set_seperation(self):
        kf = KFold(n_splits=self.cv, random_state=42, shuffle=True)
        splits = kf.split(self.x)

        self.splits = splits

    def train_model(self, test_index, train_index):
        model = clone(self.model)

        x_train = self.x.iloc[train_index]
        y_train = self.y[train_index]
        x_test = self.x.iloc[test_index]
        y_test = self.y[test_index]

        return model.fit(x_train, y_train)

    def evaluate(self, model, test_index, train_index):
        x_train = self.x.iloc[train_index]
        y_train = self.y[train_index]
        x_test = self.x.iloc[test_index]
        y_test = self.y[test_index]
        y_defaulttime_test = self.defaulttime.iloc[test_index]

        if self.loan_type is not None:
            ThreshValidation.get_loan_types(self)
            relevant_indices = list(set(x_test.index.values.tolist()) & set(self.loan_type_indices.tolist()))
            x_test['y'] = y_test
            x_test['defaulttime'] = y_defaulttime_test
            x_test = x_test.loc[relevant_indices]
            y_test = x_test['y']
            y_defaulttime_test = x_test['defaulttime']
            x_test = x_test.drop(['y','defaulttime'], axis = 1)

        probas = model.predict_proba(x_test)[:, 1]

        bt = BestThresh(x_test, y_test, probas, y_defaulttime_test)
        bt.get_best_thresh()

        return bt.best_thresh

    def parallel_fit(self):
        models = Parallel(n_jobs=self.n_jobs,
                          backend='threading')(
            delayed(ThreshValidation.train_model)(self, train_index, test_index) for train_index, test_index in
            self.splits)
        self.models = models

    def parallel_evaluation(self):
        threshes = Parallel(n_jobs=self.n_jobs,
                            backend='threading')(
            delayed(ThreshValidation.evaluate)(self, model, train_index, test_index) for
            model, (train_index, test_index) in zip(self.models, self.splits))
        self.threshes = threshes

    def get_validation_results(self):
        ThreshValidation.set_seperation(self)
        ThreshValidation.parallel_fit(self)

        ThreshValidation.set_seperation(self)
        ThreshValidation.parallel_evaluation(self)

        return self.threshes