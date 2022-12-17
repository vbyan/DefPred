from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.cluster import KMeans
from IPython.display import clear_output
from colorama import Fore
from math import ceil, floor
from sklearn.mixture import GaussianMixture as GMM
from numpy import *
import numpy as np
from pandas import *
import pandas as pd
import warnings
import time
from .filter import FilterMethods
from .bic import *
from clustering import *
from clustering import KPrototypes_model
from validclust import dunn
from sklearn.metrics.pairwise import pairwise_distances
from xgboost import XGBClassifier
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from gower import gower_matrix

warnings.filterwarnings('ignore')


def wrap(data, iterations=1, max_num_clusters=20, algorithm='KPrototypes', random_pick=1, base=[], thresh_1=1, thresh_2=0.6,
         lr=0,  stop=0.3,  corr_thresh = 0.7, variance_thresh = 0.01, significance = 0.001, dip = False):
    base_columns = []
    best_features = {}
    properties = {}
    if base:
        random_pick = 0
    filter = FilterMethods(data, base = base, corr_thresh = corr_thresh, variance_thresh = variance_thresh)
    data = filter.filter()
    base = list(set(base) & set(data.columns.values.tolist()))
    print('Number of survived columns:  ', data.shape[1])
    k = 0

    try:
        while k < iterations:
            base = list(set(data.columns.values.tolist()) & set(base))
            if base:

                base_columns = np.array(base)
                base_data = data.loc[:, base]

                if algorithm == 'KPrototypes':
                    if base_data.select_dtypes(['int64','int32','object','int']).shape[1] == base_data.shape[1]:
                        n_clusters_k, model_score_k = models_and_metrics(base_data, max_num_clusters,'KModes').metrics()
                        model = KModes_model(base_data, n_clusters_k)
                        best_method = 'KModes'

                    elif base_data.select_dtypes(['float64','float32','float']).shape[1] == base_data.shape[1]:
                        n_clusters_k, model_score_k = models_and_metrics(base_data, max_num_clusters,'KMeans').metrics()
                        model = KMeans_model(base_data, n_clusters_k)
                        best_method = 'KMeans'

                    else:
                        n_clusters_k, model_score_k = models_and_metrics(base_data, max_num_clusters,
                                                                             algorithm, dip = dip).metrics()
                        best_method = 'KPrototypes'

                        if dip:
                            pass
                        else:

                            model = KPrototypes_model(base_data, n_clusters_k)

                else:
                    n_clusters_k, model_score_k = models_and_metrics(base_data, max_num_clusters, algorithm).metrics()
                    model = KMeans_model(base_data, n_clusters_k)
                    best_method = algorithm



                best_number_of_clusters = n_clusters_k
                base_model_score = model_score_k

                model_scores = np.repeat(float('{:,.10f}'.format(base_model_score)), len(base_columns)).tolist()
                clusters = np.repeat(best_number_of_clusters, len(base_columns)).tolist()
                methods = np.repeat(best_method, len(base_columns)).tolist()
            else:
                base_columns = data.columns[random.sample(range(data.shape[1]), random_pick)].values
                base_data = data.loc[:, base_columns]
                if algorithm == 'KPrototypes':
                    if base_data.select_dtypes(['int64','int32','object','int']).shape[1] == base_data.shape[1]:
                        n_clusters_k, model_score_k = models_and_metrics(base_data, max_num_clusters,'KModes').metrics()
                        model = KModes_model(base_data, n_clusters_k)
                        best_method = 'KModes'
                    elif base_data.select_dtypes(['float64','float32','float']).shape[1] == base_data.shape[1]:
                        n_clusters_k, model_score_k = models_and_metrics(base_data, max_num_clusters,'KMeans').metrics()
                        model = KMeans_model(base_data, n_clusters_k)
                        best_method = 'KMeans'
                    else:
                        n_clusters_k, model_score_k = models_and_metrics(base_data, max_num_clusters,
                                                                             algorithm, dip = dip).metrics()
                        model = KPrototypes_model(base_data, n_clusters_k)
                        best_method = 'KPrototypes'
                else:
                    n_clusters_k, model_score_k = models_and_metrics(base_data, max_num_clusters, algorithm).metrics()
                    model = KMeans_model(base_data, n_clusters_k)
                    best_method = algorithm


                best_number_of_clusters = n_clusters_k
                base_model_score = model_score_k

                model_scores = np.repeat(float('{:,.10f}'.format(base_model_score)), len(base_columns)).tolist()
                clusters = np.repeat(best_number_of_clusters, len(base_columns)).tolist()
                methods = np.repeat(best_method, len(base_columns)).tolist()
            not_to_pick = []
            thresh_2 = np.max((thresh_2, base_model_score))
            score_thresh = thresh_2
            Not_to_pick = []
            Survivers = []
            rejected_columns = []
            rejected_scores = []

            rejected_clusters = []
            rejected_methods = []
            term = []
            stopper = 0
            counter = 0
            presort = 0
            stop = np.min((base_model_score, stop))
            print('Base model score: ',base_model_score)

            while stopper == 0:
                decider = 0
                high_level_outliers = pd.Series(Not_to_pick).value_counts()[
                    pd.Series(Not_to_pick).value_counts() >= 5].index.values
                while decider == 0:
                    try:
                        if score_thresh <= stop:
                            raise AttributeError('Too bad model from here on out')
                        if len(base_columns) + len(high_level_outliers) == data.shape[1]:
                            raise AttributeError()
                        net_new = data.drop(high_level_outliers, axis=1).drop(
                            list(set(np.append(base_columns, not_to_pick)) - set(high_level_outliers)), axis=1)
                        if presort == 0:
                            new_column = net_new.columns[np.random.randint(0, net_new.shape[1], 1)].values
                        else:
                            try:
                                sorted_list = \
                                    rejected_df[rejected_df['Column'].isin(net_new.columns.values)].sort_values(['Score'],
                                                                                                            ascending=False)[
                                    'Column'].values
                                new_column = [sorted_list[0]]
                            except IndexError:
                                raise ValueError()
                        std = data.loc[:, np.append(base_columns, new_column)] #StandardScaler().fit_transform(data.loc[:, np.append(base_columns, new_column)])
                        if 1 == 0:
                            pass
                        else:
                            if algorithm == 'KPrototypes':
                                if std.select_dtypes(['int64', 'int32', 'object', 'int']).shape[1] == std.shape[1]:
                                    mm = models_and_metrics(std, max_num_clusters, 'KModes')
                                    n_clusters_k, model_score_k = mm.metrics()

                                    model = KModes_model(std, n_clusters_k)
                                    best_method = 'KModes'
                                elif std.select_dtypes(['float64', 'float32', 'float']).shape[1] == std.shape[1]:
                                    mm = models_and_metrics(std, max_num_clusters, 'KMeans')
                                    n_clusters_k, model_score_k = mm.metrics()

                                    model = KMeans_model(std, n_clusters_k)
                                    best_method = 'KMeans'
                                else:
                                    mm = models_and_metrics(std, max_num_clusters, algorithm, dip = dip)
                                    n_clusters_k, model_score_k = mm.metrics()
                                    best_method = 'KPrototypes'

                            else:
                                mm = models_and_metrics(std, max_num_clusters,algorithm)
                                n_clusters_k, model_score_k = mm.metrics()
                                model = KMeans_model(base_data, n_clusters_k)
                                best_method = algorithm

                            model_score = model_score_k
                            best_number_of_clusters = n_clusters_k


                        if (model_score >= np.max(np.array([base_model_score, score_thresh]))):
                            base_model_score = model_score
                            base_columns = np.append(base_columns, new_column)
                            model_scores = np.append(float('{:,.10f}'.format(model_score)), model_score)
                            clusters = np.append(clusters, best_number_of_clusters)
                            methods = np.append(methods, best_method)
                            rejected_columns = []
                            rejected_scores = []
                            rejected_clusters = []
                            rejected_methods = []
                            counter = 0
                            presort = 0
                            decider = 1
                            not_to_pick = []
                            clear_output()
                            for col,  scr, cls, mts in zip(base_columns, model_scores, clusters,
                                                                methods):
                                print(Fore.GREEN + col + '  added   |  ' + 'Score: ' + str(scr) + '  |  ' + mts + ': ' + str(cls))
                            print('Not_to_pick:    ' + str(len(high_level_outliers)))
                            print('Survivers:    ' + str(len(Survivers)))
                            print('Iteration:    ' + str(k + 1))

                        else:
                            rejected_columns = np.append(rejected_columns, new_column)
                            rejected_scores = np.append(rejected_scores, float('{:,.10f}'.format(model_score)))
                            rejected_clusters = np.append(rejected_clusters, best_number_of_clusters)
                            rejected_methods = np.append(rejected_methods, best_method)
                            not_to_pick = np.append(not_to_pick, new_column)
                            counter += 1
                            print(Fore.RED + str(counter) + '. ' + new_column[
                                0] + '  rejected  |  ' +  'Score: ' + str(
                                float('{:,.10f}'.format(model_score))) + '  |  ' + best_method + ': ' + str(best_number_of_clusters)
                            )
                            pass


                    except AttributeError:
                        decider = 1
                        stopper = 1
                        k += 1
                        properties[k] = list(
                            (methods[-1], str(clusters[-1]), str(model_scores[-1])))
                        best_features[k] = base_columns.tolist()
                    except ValueError:
                        decider = 1
                        rejected_df = pd.DataFrame({'Column': rejected_columns, 'Score': rejected_scores,
                                                    'Method': rejected_methods,
                                                    'Clusters': rejected_clusters})

                        best_among_worst = \
                            rejected_df.sort_values(['Score'],
                                                                                                  ascending=False).iloc[
                            0, :]['Column']
                        base_model_score = \
                            rejected_df[rejected_df.Column == best_among_worst].Score.values.tolist()[0]

                        base_columns = np.append(base_columns, best_among_worst)

                        model_scores = np.append(model_scores, float('{:,.10f}'.format(base_model_score)))
                        clusters = np.append(clusters,
                            rejected_df[rejected_df.Column == best_among_worst].Clusters.values.tolist()[0])
                        methods = np.append(methods, rejected_df[
                            rejected_df.Column == best_among_worst].Method.values.tolist()[0])
                        score_thresh = base_model_score - lr
                        print(Fore.GREEN + 'Score_thresh = ' + str(score_thresh))
                        time.sleep(3)
                        not_to_pick = []
                        rejected_columns = []
                        rejected_scores = []

                        rejected_clusters = []
                        rejected_methods = []
                        clear_output()
                        counter = 0
                        presort = 1
                        print(base_columns)
                        print(model_scores)
                        print(clusters)
                        print(methods)
                        time.sleep(10)
                        for col,  scr, cls, mts in zip(base_columns,  model_scores, clusters,methods):
                            print(Fore.GREEN + col + '  added  |  ' + 'Score: ' + str(scr) + '  |  ' + mts + ': ' + str(cls))
                        print('Not_to_pick:    ' + str(len(high_level_outliers)))
                        print('Survivers:    ' + str(len(Survivers)))
                        print('Iteration:    ' + str(k + 1))

        clear_output()
        return best_features, properties
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        best_features[k] = base_columns.tolist()
        properties[k] = list((methods[-1], str(clusters[-1]), str(model_scores[-1])))
        return best_features, properties


class BackElim():
    def __init__(self):
        pass

    def back_direct(self, data, max_num_clusters=20, algorithm='GMM', lr=0.02):
        base_data = data #StandardScaler().fit_transform(data)
        if algorithm == 'KMeans':
            n_clusters_k, model_score_k= models_and_metrics(base_data, max_num_clusters, 'KMeans').metrics()
            model_score_g = -2
        else:
            n_clusters_g, model_score_g = models_and_metrics(base_data, max_num_clusters, 'GMM').metrics()
            model_score_k = -2
        if model_score_k > model_score_g:
            best_method = 'KMeans'
            best_number_of_clusters = n_clusters_k
            base_model_score = model_score_k
            model = KMeans(n_clusters=n_clusters_k, random_state=42).fit(base_data)
            labels = model.labels_
        else:
            best_method = 'GMM'
            best_number_of_clusters = n_clusters_g
            base_model_score = model_score_g
            model = GMM(n_clusters_g, covariance_type='full', random_state=42).fit(base_data)
            labels = model.predict(base_data)
        base_distribution = pd.Series(labels).value_counts().values.std() / pd.Series(
            labels).value_counts().values.mean()
        new = data
        decider = 0
        runner = 0
        counter = 0
        not_to_pick = []
        try:
            while True:
                decider = 0
                try:
                    while decider == 0:
                        counter += 1
                        runner += 1
                        new_column = new.drop(not_to_pick, axis=1).iloc[:,
                                     np.random.randint(0, new.drop(not_to_pick, axis=1).shape[1], 1)].columns.values[0]
                        new_data = new.drop(new_column, axis=1) #StandardScaler().fit_transform(new.drop(new_column, axis=1))
                        if algorithm == 'KMeans':
                            n_clusters_k, model_score_k= models_and_metrics(new_data, max_num_clusters,
                                                                                'KMeans').metrics()
                            model_score_g = -2
                        else:
                            n_clusters_g, model_score_g= models_and_metrics(new_data, max_num_clusters,
                                                                                'GMM').metrics()
                            model_score_k = -2
                        if model_score_k > model_score_g:
                            best_method = 'KMeans'
                            best_number_of_clusters = n_clusters_k
                            model_score = model_score_k
                            model = KMeans(n_clusters=n_clusters_k, random_state=42).fit(new_data)
                            labels = model.labels_
                        else:
                            best_method = 'GMM'
                            best_number_of_clusters = n_clusters_g
                            model_score = model_score_g
                            model = GMM(n_clusters_g, covariance_type='full', random_state=42).fit(new_data)
                            labels = model.predict(new_data)

                        if (model_score >= base_model_score + lr) :
                            print(
                                Fore.YELLOW + str(runner) + '. ' + new_column + '     Eliminated' + '    score  ' + str(
                                    "%+g" % (round(model_score - base_model_score, 3))) + '   Clusters:  ' + str(
                                    best_number_of_clusters))
                            base_model_score = model_score
                            base_distribution = distribution
                            new = new.drop(new_column, axis=1)
                            decider = 1
                            if counter == 30:
                                print(Fore.BLUE + 'Score:  ' + str(model_score) )
                                counter = 0
                        else:
                            print(Fore.GREEN + str(runner) + '. ' + new_column + '     Survived' + '    score  ' + str(
                                "%+g" % (round(model_score - base_model_score, 3)))  + '   Clusters:  ' + str(
                                best_number_of_clusters))

                            not_to_pick = np.append(not_to_pick, new_column)
                except ValueError:
                    self.new = new
                    self.score = base_model_score
                    return self.new, self.score
        except KeyboardInterrupt:
            self.new = new
            self.score = base_model_score
            return self.new, self.score

    def iterator(self, data, max_num_clusters=20, algorithm='GMM', lr=0.02):
        looper,score = BackElim().back_direct(data, max_num_clusters, algorithm, lr=lr)
        if data.shape[1] != looper.shape[1]:
            while True:
                temp = looper
                clear_output()
                looper, score = BackElim().back_direct(looper, max_num_clusters, algorithm, lr=lr)
                if temp.shape[1] == looper.shape[1]:
                    return looper.columns.values
                    break
                else:
                    continue
        return looper.columns.values

    def bi_direct(self, data, base_columns, base_model_score, base_distribution, Survivers, max_num_clusters,
                  algorithm, cruelty,elim):
        set = base_columns
        for i in Survivers:
            set = np.delete(set, np.where(set == i))
        set = set.tolist()
        if len(set) >= cruelty + 1 - elim:
            outsider = set[np.random.randint(0, ceil(len(set) * 0.7), 1)[0]]
            std = data.loc[:, base_columns].drop(outsider, axis=1) #StandardScaler().fit_transform(data.loc[:, base_columns].drop(outsider, axis=1))
            out = None
            surviver = None
            if algorithm == 'KMeans':
                n_clusters_k, model_score_k = models_and_metrics(std, max_num_clusters, 'KMeans').metrics()
                model_score_g = -2
            else:
                n_clusters_g, model_score_g = models_and_metrics(std, max_num_clusters, 'GMM').metrics()
                model_score_k = -2
            if model_score_k > model_score_g:
                best_method = 'KMeans'
                model_score = model_score_k
                best_number_of_clusters = n_clusters_k
                model = KMeans(n_clusters=n_clusters_k, random_state=42).fit(std)
                labels = model.labels_
            else:
                best_method = 'GMM'
                model_score = model_score_g
                best_number_of_clusters = n_clusters_g
                model = GMM(n_clusters_g, covariance_type='full', random_state=42).fit(std)
                labels = model.predict(std)
            distribution = pd.Series(labels).value_counts().values.std() / pd.Series(
                labels).value_counts().values.mean()
            if (model_score >= base_model_score) & (distribution < base_distribution) :
                out = outsider
            else:
                surviver = outsider
                pass
            return out, surviver
        else:
            return None, None


class models_and_metrics:
    def __init__(self, std, num_of_clusters, method, dip = False):
        self.std = std
        self.num_of_clusters = num_of_clusters + 1
        self.method = method
        self.dip = dip

    def models(self):
        if self.dip:
            pass

        else:
            K = range(2, self.num_of_clusters)
            if self.method == 'KPrototypes':
                categorical_indices = locate_categoricals(self.std)
                models = [KPrototypes(n_clusters=k, random_state=42, n_jobs=-1, init='Huang', gamma=100, n_init=10).fit(
                    self.std, categorical=categorical_indices) for k in K]
                metric = 'gower'
                weight = weights_for_gower(self.std)
                dist = gower_matrix(self.std.values, weight=weight)

            elif self.method == 'KModes':
                models = [KModes(n_clusters=k, random_state=42, n_jobs=-1).fit(self.std) for k in K]
                metric = 'hamming'
                dist = pairwise_distances(self.std, metric=metric)

            elif self.method == 'KMeans':
                models = [KMeans(n_clusters=k, random_state=42, n_jobs=-1).fit(self.std) for k in K]
                metric = 'euclidean'
                dist = pairwise_distances(self.std, metric=metric)
            np.fill_diagonal(dist, 0)
            calinski = [calinski_harabasz_score(self.std, model.labels_) for model in models]
            david = [davies_bouldin_score(self.std, model.labels_, metric=metric) for model in models]
            david = pd.Series(david)
            david[np.isposinf(david)] = 1e+10
            david = david.values.tolist()
            Dunn = [dunn(dist, model.labels_) for model in models]
            Dunn = pd.Series(Dunn)
            Dunn[np.isposinf(Dunn)] = 1
            Dunn = Dunn.values.tolist()
            silh = [silhouette_score(dist, model.labels_, metric='precomputed') for model in models]
            self.david = (np.max(david) / np.array(david)).tolist()
            self.calinski = calinski
            self.dunn = Dunn
            self.silh = silh
            self.models = models
            self.dist = dist

    def metrics(self):
        if self.dip:
            best_number_of_clusters = 'dip'
            weight = weights_for_gower(self.std)
            dist = gower_matrix(self.std.values, weight=weight)
            model_score = (dip(dist, num_bins = 10) + dip(dist, num_bins = 30)+ dip(dist, num_bins = 100))/3
        else:
            models_and_metrics.models(self)
            scores = {}
            scores['david'] = self.david
            scores['dunn'] = self.dunn
            scores['calinski'] = self.calinski
            scores['silh'] = self.silh
            scores = pd.DataFrame(scores)
            scores = pd.DataFrame(MinMaxScaler().fit_transform(scores), index=np.arange(2, self.num_of_clusters),
                                  columns=scores.columns)

            scores['silh'] = 2 * scores['silh']
            scores['dunn'] = 2 * scores['dunn']

            scores['final'] = scores.sum(axis=1)
            pretendents = scores['final'].sort_values(ascending=False).index.values.tolist()[:3]
            for pretendent in pretendents:
                model = self.models[pretendent - 2]
                distribution = pd.Series(model.labels_).value_counts().std() / pd.Series(
                    model.labels_).value_counts().mean() * self.num_of_clusters ** (1 / 10) / pretendent ** (1 / 10)
                if distribution <= 5:
                    best_number_of_clusters = pretendent
                    break
                else:
                    pass
                if pretendent == pretendents[-1]:
                    best_number_of_clusters = pretendents[0]
            best_model = self.models[best_number_of_clusters - 2]
            calinski_to_silh = MinMaxScaler((np.min(self.silh) - 0.000001, np.max(self.silh))).fit_transform(
                scores['calinski'].values.reshape(-1, 1)).reshape(-1).tolist()
            dunn_to_silh = MinMaxScaler((np.min(self.silh) - 0.000001, np.max(self.silh))).fit_transform(
                scores['dunn'].values.reshape(-1, 1)).reshape(-1).tolist()
            david_to_silh = MinMaxScaler((np.min(self.silh) - 0.000001, np.max(self.silh))).fit_transform(
                scores['david'].values.reshape(-1, 1)).reshape(-1).tolist()
            model_score = silhouette_score(self.dist, best_model.labels_, metric='precomputed') * 0.4 + \
                          calinski_to_silh[
                              best_number_of_clusters - 2] * 0.2 + david_to_silh[best_number_of_clusters - 2] * 0.2 + \
                          dunn_to_silh[
                              best_number_of_clusters - 2] * 0.2
            self.pretendents = pretendents

        return best_number_of_clusters, model_score

    @property
    def get_pretendents_(self):
        return self.pretendents

def locate_categoricals(data):
    categorical_indices = []
    categorical_columns = data.select_dtypes(['int', 'int64', 'object', 'int32']).columns.values
    for column in categorical_columns:
        categorical_indices += [np.where(data.columns.values == column)[0][0]]

    return categorical_indices


def xgb_validation(data, algorithm, base, num_clusters):
    if data.select_dtypes(['int64', 'int32', 'object', 'int']).shape[1] == data.shape[1]:
        model = KModes_model(data,num_clusters)
    elif data.select_dtypes(['float64', 'float32', 'float']).shape[1] == data.shape[1]:
        model = KMeans_model(data,num_clusters)
    else:
        model = KPrototypes_model(data, num_clusters, init = 'Huang', n_init = 10, gamma = 100)
    shape = data.shape[1]
    labels = model.labels_
    x = pd.DataFrame(data.values, columns = data.columns)
    try:
        x[x.select_dtypes('O').columns] = x.select_dtypes('O').astype('int')
    except:
        pass
    y = labels
    model = XGBClassifier(n_estimators=100, alpha=5, max_depth=10, seed=12345, min_child_weight=3,
                              colsample_bytree=0.3, learning_rate=0.05, silent=True, verbosity=0)
    model.fit(x,y)
    importance1 = float('{:,.2f}'.format(pd.Series(model.feature_importances_, index = data.columns)[base].sum()))
    importance2 = model.feature_importances_[-1]
    if (importance1 <= 0.5 - shape*0.01) | (importance1 >= 0.95)|(importance2 < 0.01):
         verdict = 'reject'
    else:
        verdict = 'approve'
    return importance1,verdict


def large_set(data, max_num_clusters=20, iterations=10, thresh_1=1, thresh_2=0.6, base=[], Backelim=True):
    best_features, _ = wrap(data, iterations=iterations, max_num_clusters=max_num_clusters, base=base,
                            thresh_1=thresh_1, thresh_2=thresh_2)
    all = {}
    comb = {}
    for i in list(best_features.keys()):
        for k in np.delete(list(best_features.keys()), i):
            comb[str(i) + '_' + str(k)] = list(set(best_features[k]) & set(best_features[i]))
        all[str(i)] = comb
    Base = base
    for i in all.keys():
        for j in all[i].keys():
            Base = set(Base) | set(all[i][j])
    final = list(Base)
    model_score = 0
    distribution = 10
    return best_features, final


def row_selection(data, thresh=100):
    import clustering
    iteration = 0
    scaler = StandardScaler().fit(data)
    while True:
        std = scaler.transform(data)
        iteration += 1
        clusters, _, __ = models_and_metrics(std, 30, 'GMM').metrics()
        model, _ = clustering.GMM_model(data, clusters)
        labels = model.predict(std)
        distribution = pd.Series(labels).value_counts()
        if distribution[distribution < thresh].values.tolist():
            data['Cluster'] = labels
            out = distribution[distribution < thresh].index.values
            clear_output()
            clustering.decomposition(data.drop('Cluster', axis=1), 100, labels, method='TSNE')
            print('Iteration:  ' + str(iteration))
            data = data[~data['Cluster'].isin(out)]
            data = data.drop('Cluster', axis=1)
        else:
            clear_output()
            clustering.decomposition(data, 100, labels, method='TSNE')
            print('Iteration:  ' + str(iteration))
            return data
            break

def dip(samples, num_bins=1000, p=0.95, table=False):

    samples = samples / np.abs(samples).max()
    pdf, idxs = np.histogram(samples, bins=num_bins)
    idxs = idxs[:-1] + np.diff(idxs)
    pdf  = pdf / pdf.sum()

    cdf = np.cumsum(pdf, dtype=float)
    assert np.abs(cdf[-1] - 1) < 1e-3

    D = 0
    ans = 0
    check = False
    while True:
        gcm_values, gcm_contact_points   = gcm_cal(cdf, idxs)
        lcm_values, lcm_contact_points = lcm_cal(cdf, idxs)

        d_gcm, gcm_diff = sup_diff(gcm_values, lcm_values, gcm_contact_points)
        d_lcm, lcm_diff = sup_diff(gcm_values, lcm_values, lcm_contact_points)

        if d_gcm > d_lcm:
            xl = gcm_contact_points[d_gcm == gcm_diff][0]
            xr = lcm_contact_points[lcm_contact_points >= xl][0]
            d  = d_gcm
        else:
            xr = lcm_contact_points[d_lcm == lcm_diff][-1]
            xl = gcm_contact_points[gcm_contact_points <= xr][-1]
            d  = d_lcm

        gcm_diff_ranged = np.abs(gcm_values[:xl+1] - cdf[:xl+1]).max()
        lcm_diff_ranged = np.abs(lcm_values[xr:]  - cdf[xr:]).max()

        if d <= D or xr == 0 or xl == cdf.size:
            ans = D
            break
        else:
            D = max(D, gcm_diff_ranged, lcm_diff_ranged)

        cdf = cdf[xl:xr+1]
        idxs = idxs[xl:xr+1]
        pdf = pdf[xl:xr+1]

    if table:
        p_threshold = p_table(p, samples.size, 10000)
        if ans < p_threshold:
            check = True
        return ans, check

    return ans


def gcm_cal(cdf, idxs):
    local_cdf = np.copy(cdf)
    local_idxs = np.copy(idxs)
    gcm = [local_cdf[0]]
    contact_points = [0]
    while local_cdf.size > 1:
        distances = local_idxs[1:] - local_idxs[0]
        slopes = (local_cdf[1:] - local_cdf[0]) / distances
        slope_min = slopes.min()
        slope_min_idx = np.where(slopes == slope_min)[0][0] + 1
        gcm.append(local_cdf[0] + distances[:slope_min_idx] * slope_min)
        contact_points.append(contact_points[-1] + slope_min_idx)
        local_cdf = local_cdf[slope_min_idx:]
        local_idxs = local_idxs[slope_min_idx:]
    return np.hstack(gcm), np.hstack(contact_points)


def lcm_cal(cdf, idxs):
    values, points = gcm_cal(1-cdf[::-1], idxs.max() - idxs[::-1])
    return 1 - values[::-1], idxs.size - points[::-1] - 1


def sup_diff(alpha, beta, contact_points):
    diff = np.abs((alpha[contact_points] - beta[contact_points]))
    return diff.max(), diff


def p_table(p, sample_size, n_samples):
    data = [np.random.randn(sample_size) for _ in range(n_samples)]
    dips = np.hstack([dip(samples, table=False) for samples in data])
    return np.percentile(dips, p*100)

def dip_test(data):
    if data.select_dtypes(['int64', 'int32', 'object', 'int']).shape[1] == data.shape[1]:
        dist = pairwise_distances(data, metric = 'hamming')
    elif data.select_dtypes(['float64', 'float32', 'float']).shape[1] == data.shape[1]:
        dist = pairwise_distances(data, metric='euclidean')
    else:
        weight = weights_for_gower(data)
        dist = gower_matrix(data.values, weight = weight)
    test_result = float('{:,.9f}'.format(dip(dist, num_bins = int(data.shape[0]/10))))
    return test_result

def weights_for_gower(data):
    categorical_indices = locate_categoricals(data)
    cat_weight = data.drop(data.iloc[:,categorical_indices].columns, axis = 1).values.reshape(-1).std() * 0.5
    base_array = np.repeat(1,data.shape[1]).astype('float32')
    base_array[categorical_indices] = cat_weight
    weights = base_array
    return weights