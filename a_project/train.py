import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from helpers import class_weights
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve

#This is a generic algorithm to find the best model
#Includes 3 types of models XGBoost, CatBoost and RandomForest
#includes hyperparameter tuning  using GridSearchCV
#includes also model combinations such as voting and stacking
#Cross-validation is used in all stages of model-evaluation
#Can be used for Classification and Regression tasks

class Train():
    def __init__(self, task, cv=5, cat_lr=0.05, xgb_lr=0.02, scoring='roc_auc',refit = 'auc', n_jobs=-1, only_cat = False, calibrate = False):
        self.task = task
        self.cv = cv
        self.cat_lr = cat_lr
        self.xgb_lr = xgb_lr
        self.scoring = scoring
        self.refit = refit
        self.n_jobs = n_jobs
        self.only_cat = only_cat
        self.calibrate = calibrate

    def models_and_grids(self, weights):
        catboost_grid = {'max_depth': [6, 7, 8, 10], 'n_estimators': [50, 100, 300, 500, 1000]}
        xgboost_grid = {'max_depth': [6, 8, 10, 12, 15], 'n_estimators': [50, 100, 200, 300, 500],
                        'colsample_bytree': [0.3, 0.7, 1]}
        rf_grid = {'max_depth': [6, 8, 10, 12, 15, 20], 'min_samples_split': [10, 20, 30, 40],
                   'min_samples_leaf': [5, 10]}

        if self.task == 'classification':
            catboost_model = CatBoostClassifier(random_state=116, cat_features=self.cat_features, class_weights=weights,
                                                learning_rate=self.cat_lr, thread_count=self.n_jobs)
            if np.unique(self.y_train).size > 2:
                 xgboost_model = XGBClassifier(seed=116, learning_rate=self.xgb_lr,
                                          n_jobs=self.n_jobs)
            else:
                 xgboost_model = XGBClassifier(seed=116, scale_pos_weight=weights[1] / weights[0], learning_rate=self.xgb_lr,
                                          n_jobs=self.n_jobs)
            rf_model = RandomForestClassifier(random_state=116, class_weight='balanced', n_estimators=300,
                                              n_jobs=self.n_jobs)
        elif self.task == 'regression':
            catboost_model = CatBoostRegressor(random_state=116,cat_features=self.cat_features,
                                                learning_rate=self.cat_lr, thread_count=self.n_jobs)
            xgboost_model = XGBRegressor(seed=116, learning_rate=self.xgb_lr,
                                          n_jobs=self.n_jobs)
            rf_model = RandomForestRegressor(random_state=116, n_estimators=300,
                                              n_jobs=self.n_jobs)
        else:
            raise ValueError('Wrong task type')


        self.catboost_grid = catboost_grid
        self.xgboost_grid = xgboost_grid
        self.rf_grid = rf_grid
        self.catboost_model = catboost_model
        self.xgboost_model = xgboost_model
        self.rf_model = rf_model

    def gridsearch(self):
        if self.only_cat:
            grid_cat = GridSearchCV(estimator=self.catboost_model, param_grid=self.catboost_grid, scoring=self.scoring,
                                    refit=self.refit,
                                    cv=self.cv, n_jobs=self.n_jobs, verbose=10)
            grid_cat.fit(self.x_train, self.y_train)
            self.grid_cat = grid_cat
        else:
            grid_cat = GridSearchCV(estimator=self.catboost_model, param_grid=self.catboost_grid, scoring=self.scoring,
                                    refit=self.refit,
                                    cv=self.cv, n_jobs=self.n_jobs, verbose=10)

            grid_xgb = GridSearchCV(estimator=self.xgboost_model, param_grid=self.xgboost_grid, scoring=self.scoring,
                                    refit=self.refit, cv=self.cv,
                                    n_jobs=self.n_jobs, verbose=10)
            grid_rf = GridSearchCV(estimator=self.rf_model, param_grid=self.rf_grid, scoring=self.scoring,
                                   refit=self.refit,
                                   cv=self.cv, n_jobs=self.n_jobs,
                                   verbose=10)


            grid_cat.fit(self.x_train, self.y_train)
            if np.unique(self.y_train).size > 2:
                sample_weight = compute_sample_weight(class_weight='balanced', y=self.y_train)
                grid_xgb.fit(self.x_train, self.y_train,sample_weight = sample_weight)
            else:
                grid_xgb.fit(self.x_train, self.y_train)
            grid_rf.fit(self.x_train, self.y_train)

            self.grid_cat = grid_cat
            self.grid_xgb = grid_xgb
            self.grid_rf = grid_rf

    def voting(self):
        estimators = [('cat', self.grid_cat.best_estimator_), ('rf', self.grid_rf.best_estimator_),
                      ('xgb', self.grid_xgb.best_estimator_)]

        if self.task == 'classification':
            voter = VotingClassifier(estimators=estimators, voting='soft', n_jobs=self.n_jobs)
        else:
            voter = VotingRegressor(estimators=estimators, n_jobs=self.n_jobs)

        voter.fit(self.x_train, self.y_train)

        voting_cvs = cross_val_score(voter, self.x_train, self.y_train, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs)

        self.voter = voter
        self.voting_cvs = voting_cvs

    def stacking(self):
        estimators = [('cat', self.grid_cat.best_estimator_), ('rf', self.grid_rf.best_estimator_),
                      ('xgb', self.grid_xgb.best_estimator_)]

        if self.task == 'classification':
            stacker = StackingClassifier(estimators=estimators,
                                     final_estimator=LogisticRegression(class_weight='balanced'), cv=self.cv)
        else:
            stacker = StackingRegressor(estimators=estimators,
                                         final_estimator=RandomForestRegressor(n_estimators = 100, random_state = 42), cv=self.cv
                                         )
        try:
            stacker.fit(self.x_train, self.y_train)
            stacking_cvs = cross_val_score(stacker, self.x_train, self.y_train, cv=self.cv, scoring=self.scoring)
        except:
            stacker = self.grid_cat.best_estimator_
            stacking_cvs = self.grid_cat.best_score_

        self.stacker = stacker
        self.stacking_cvs = stacking_cvs

    def calibration(self):
        if self.x_train.shape[0] < 1000:
            method = 'sigmoid'
        else:
            method = 'isotonic'
        calibrator = Calibrate(self.best_model, self.x_train, self.y_train, method = method)
        cc = calibrator.get_cc()

        self.cc = cc

    def best_model(self):
        if self.only_cat:
            self.best_model = self.grid_cat.best_estimator_
            self.validation = self.grid_cat.best_score_
        else:
            models = [self.grid_cat.best_estimator_, self.grid_xgb.best_estimator_,
                      self.grid_rf.best_estimator_, self.voter, self.stacker]
            val_results = [self.grid_cat.best_score_, self.grid_xgb.best_score_, self.grid_rf.best_score_,
                           np.mean(self.voting_cvs), np.mean(self.stacking_cvs)]
            best_index = np.array(val_results).argmax()
            best_model = models[best_index]

            validation = {}
            for model_name, validation_result in zip(['cat', 'xgb', 'rf', 'voting', 'stacking'], val_results):
                validation[model_name] = validation_result

            self.best_model = best_model
            self.validation = validation



    def fit(self,x_train, y_train, cat_features = None):
        self.x_train = x_train
        self.y_train = y_train
        self.cat_features = cat_features

        if self.task == 'classification':
            weights = class_weights(y_train)
        else:
            weights = None

        Train.models_and_grids(self, weights)
        if self.only_cat:
            Train.gridsearch(self)
        else:
            Train.gridsearch(self)
            Train.voting(self)
            Train.stacking(self)

        Train.best_model(self)

        if self.calibrate:
            Train.calibration(self)



    @property
    def validation_results_(self):
        return self.validation

    @property
    def best_model_(self):
        return self.best_model

    @property
    def cc_(self):
        return self.cc





class Calibrate:
    def __init__(self, model, x_train, y_train, cv=5, n_jobs=-1, method = 'isotonic'):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.cv = cv
        self.n_jobs = n_jobs
        self.method = method

    def get_cc(self):
        base_estimator = clone(self.model)
        cc = CalibratedClassifierCV(base_estimator=base_estimator, cv=self.cv, n_jobs=self.n_jobs, method= self.method)
        cc.fit(self.x_train, self.y_train)
        self.cc = cc
        return cc

    def curves(self, x_test, y_test, n_bins=20):
        pred1 = self.model.predict_proba(x_test)[:, 1]
        pred2 = self.cc.predict_proba(x_test)[:, 1]
        prob_true1, prob_pred1 = calibration_curve(y_test,
                                                   pred1)
        prob_true2, prob_pred2 = calibration_curve(y_test,
                                                   pred2)

        plt.figure(figsize=(10, 5))
        plt.plot(prob_pred1, prob_true1, c='r')
        plt.plot(prob_pred2, prob_true2, c='g')
        plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))


def remove_temporary_attrs(obj, leave):
    all_attrs = list(obj.__dict__.keys())

    for attr in leave:
        if attr in all_attrs:
            all_attrs.remove(attr)

    for attr in all_attrs:
        delattr(obj, attr)




