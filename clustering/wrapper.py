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
from validclust import dunn
from sklearn.metrics.pairwise import pairwise_distances
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')


def wrap(data, iterations=1, max_num_clusters=20, algorithm='GMM', random_pick=1, base=[], thresh_1=1, thresh_2=0.6,
         lr=0.02, backelim=True, stop=0.3, cruelty=1, preelim = False, significance = 0.01,corr_thresh = 0.7, variance_thresh = 0.1):
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
    def BE_process():
        nonlocal backelim;nonlocal base_columns;nonlocal cruelty;nonlocal data;nonlocal base_model_score
        nonlocal base_distribution;nonlocal Survivers;nonlocal max_num_clusters;nonlocal algorithm;nonlocal term;nonlocal model_scores
        nonlocal clusters;nonlocal methods;nonlocal distributions; nonlocal elim
        if backelim:
            out, surviver = BackElim().bi_direct(data, base_columns, base_model_score,
                                                 base_distribution,
                                                 Survivers, max_num_clusters, algorithm,cruelty,elim)
            term = (np.array(term) + 1).tolist()
            if surviver != None:
                Survivers += [surviver]
                term += [1]
                Survivers = np.delete(Survivers, np.where(np.array(term) == cruelty)).tolist()
                term = np.delete(term, np.where(np.array(term) ==  cruelty)).tolist()
                print(Fore.CYAN + surviver + '   Survived')
            if out != None:
                model_scores = np.delete(model_scores, np.where(base_columns == out))
                clusters = np.delete(clusters, np.where(base_columns == out))
                methods = np.delete(methods, np.where(base_columns == out))
                distributions = np.delete(distributions, np.where(base_columns == out))
                base_columns = np.delete(base_columns, np.where(base_columns == out))
                print(Fore.YELLOW + out + '   Eliminated')
            if (surviver == None)&(out ==None):
                return 'break'



    try:
        while k < iterations:
            base = list(set(data.columns.values.tolist()) & set(base))
            if base:
                if preelim:
                    base_columns = BackElim().iterator(data.loc[:, base], algorithm = algorithm)
                    clear_output()
                    base_data = data.loc[:, base_columns.tolist()] #StandardScaler().fit_transform(data.loc[:, base_columns.tolist()])
                else:
                    base_columns = np.array(base)
                    base_data = data.loc[:, base] #StandardScaler().fit_transform(data.loc[:, base])

                if algorithm == 'KMeans':
                    n_clusters_k, model_score_k= models_and_metrics(base_data, max_num_clusters, 'KMeans').metrics()
                    model_score_g = -2
                else:
                    n_clusters_g, model_score_g= models_and_metrics(base_data, max_num_clusters, 'GMM').metrics()
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
                    labels).value_counts().values.mean() * max_num_clusters ** (1/10) / best_number_of_clusters ** (1/10)
                distributions = np.repeat(base_distribution, len(base_columns)).tolist()
                model_scores = np.repeat(base_model_score, len(base_columns)).tolist()
                clusters = np.repeat(best_number_of_clusters, len(base_columns)).tolist()
                methods = np.repeat(best_method, len(base_columns)).tolist()
            else:
                base_columns = data.columns[random.sample(range(data.shape[1]), random_pick)].values
                base_data = data.loc[:, base_columns] #StandardScaler().fit_transform(data.loc[:, base_columns])
                if algorithm == 'KMeans':
                    n_clusters_k, model_score_k = models_and_metrics(base_data, max_num_clusters, 'KMeans').metrics()
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
                    labels).value_counts().values.mean() * max_num_clusters ** (1/10) / best_number_of_clusters ** (1/10)
                distributions = np.repeat(base_distribution, len(base_columns)).tolist()
                model_scores = np.repeat(base_model_score, len(base_columns)).tolist()
                clusters = np.repeat(best_number_of_clusters, len(base_columns)).tolist()
                methods = np.repeat(best_method, len(base_columns)).tolist()
            not_to_pick = []
            thresh_2 = np.max((thresh_2, base_model_score))
            distr_thresh = thresh_1
            score_thresh = thresh_2
            Not_to_pick = []
            Survivers = []
            rejected_columns = []
            rejected_scores = []
            rejected_distributions = []
            rejected_clusters = []
            rejected_methods = []
            term = []
            stopper = 0
            counter = 0
            presort = 0
            stop = np.min((base_model_score, stop))
            print('Base model score: ',base_model_score)
            print('Base distribution: ',base_distribution)
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
                        if dip(pairwise_distances(std), table = False) >= significance:
                            if algorithm == 'KMeans':
                                mm = models_and_metrics(std, max_num_clusters, 'KMeans')
                                n_clusters_k, model_score_k = mm.metrics()
                                pretendents = mm.get_pretendents_
                                model_score_g = -2  # unreachable low number for silhouette_score
                            else:
                                mm = models_and_metrics(std, max_num_clusters, 'GMM')
                                n_clusters_g, model_score_g = mm.metrics()
                                pretendents = mm.get_pretendents_
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
                                labels).value_counts().values.mean() * max_num_clusters ** (
                                                       1 / 10) / best_number_of_clusters ** (1 / 10)
                        else:
                            model_score = -1
                            best_number_of_clusters = 1
                            distribution = 0
                            best_method = "Doesn't matter"

                        if (model_score >= np.min(np.array([base_model_score, score_thresh]))) & (
                                distribution < np.max(np.array([base_distribution, distr_thresh]))):
                            base_model_score = model_score
                            base_distribution = distribution
                            base_columns = np.append(base_columns, new_column)
                            distributions = np.append(distributions, distribution)
                            model_scores = np.append(model_scores, model_score)
                            clusters = np.append(clusters, best_number_of_clusters)
                            methods = np.append(methods, best_method)
                            rejected_columns = []
                            rejected_scores = []
                            rejected_distributions = []
                            rejected_clusters = []
                            rejected_methods = []
                            counter = 0
                            presort = 0
                            decider = 1
                            not_to_pick = []
                            clear_output()
                            for col, dist, scr, cls, mts in zip(base_columns, distributions, model_scores, clusters,
                                                                methods):
                                print(Fore.GREEN + col + '  added     |   ' + 'Distribution: ' + str(
                                    dist) + '      |    Score: ' + str(scr) + '    |     ' + mts + ': ' + str(cls))
                            print('Not_to_pick:    ' + str(len(high_level_outliers)))
                            print('Survivers:    ' + str(len(Survivers)))
                            print('Iteration:    ' + str(k + 1))
                            for elim in range(cruelty):
                                decision = BE_process()
                                if decision =='break':
                                    break
                        else:
                            rejected_columns = np.append(rejected_columns, new_column)
                            rejected_scores = np.append(rejected_scores, model_score)
                            rejected_distributions = np.append(rejected_distributions, distribution)
                            rejected_clusters = np.append(rejected_clusters, best_number_of_clusters)
                            rejected_methods = np.append(rejected_methods, best_method)
                            not_to_pick = np.append(not_to_pick, new_column)
                            counter += 1
                            if (model_score < 0) | (distribution > distr_thresh + 2):
                                for i in range(1):
                                    Not_to_pick += [new_column[0]]
                            else:
                                pass
                            print(Fore.RED + str(counter) + '. ' + new_column[
                                0] + '  rejected     |    ' + 'Distribution: ' + str(
                                distribution) + '    |    Score: ' + str(
                                model_score) + '    |     ' + best_method + ': ' + str(best_number_of_clusters))
                            pass
                    except AttributeError:
                        decider = 1
                        stopper = 1
                        k += 1
                        properties[k] = list(
                            (methods[-1], str(clusters[-1]), str(model_scores[-1]), str(distributions[-1])))
                        best_features[k] = base_columns.tolist()
                    except ValueError:
                        decider = 1
                        rejected_df = pd.DataFrame({'Column': rejected_columns, 'Score': rejected_scores,
                                                    'Distribution': rejected_distributions, 'Method': rejected_methods,
                                                    'Clusters': rejected_clusters})
                        if rejected_df[rejected_df['Distribution'] < distr_thresh].shape[0]:
                            best_among_worst = \
                            rejected_df[(rejected_df['Distribution'] < distr_thresh)].sort_values(['Score'],
                                                                                                  ascending=False).iloc[
                            0,:]['Column']
                            base_model_score = \
                            rejected_df[rejected_df.Column == best_among_worst].Score.values.tolist()[0]
                            base_distribution = \
                            rejected_df[rejected_df.Column == best_among_worst].Distribution.values.tolist()[0]
                            base_columns = np.append(base_columns, best_among_worst)
                            distributions = np.append(distributions, base_distribution)
                            model_scores = np.append(model_scores, base_model_score)
                            clusters = np.append(clusters, int(
                                rejected_df[rejected_df.Column == best_among_worst].Clusters.values.tolist()[0]))
                            methods = np.append(methods, rejected_df[
                                rejected_df.Column == best_among_worst].Method.values.tolist()[0])
                        distr_thresh = distr_thresh + lr
                        score_thresh = base_model_score - lr
                        print(Fore.GREEN + 'Score_thresh = ' + str(score_thresh))
                        print(Fore.GREEN + 'Distr_thresh = ' + str(distr_thresh))
                        time.sleep(3)
                        not_to_pick = []
                        rejected_columns = []
                        rejected_scores = []
                        rejected_distributions = []
                        rejected_clusters = []
                        rejected_methods = []
                        clear_output()
                        counter = 0
                        presort = 1
                        for col, dist, scr, cls, mts in zip(base_columns, distributions, model_scores, clusters,
                                                            methods):
                            print(Fore.GREEN + col + '  added     |   ' + 'Distribution: ' + str(
                                dist) + '      |    Score: ' + str(scr) + '    |     ' + mts + ': ' + str(cls))
                        print('Not_to_pick:    ' + str(len(high_level_outliers)))
                        print('Survivers:    ' + str(len(Survivers)))
                        print('Iteration:    ' + str(k + 1))
                        for elim in range(cruelty):
                            decision = BE_process()
                            if decision == 'break':
                                break
        clear_output()
        return best_features, properties
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        best_features[k] = base_columns.tolist()
        properties[k] = list((methods[-1], str(clusters[-1]), str(model_scores[-1]), str(distributions[-1])))
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
                        distribution = pd.Series(labels).value_counts().values.std() / pd.Series(
                            labels).value_counts().values.mean()
                        if (distribution <= base_distribution + lr) & (model_score >= base_model_score + lr) :
                            print(
                                Fore.YELLOW + str(runner) + '. ' + new_column + '     Eliminated' + '    score  ' + str(
                                    "%+g" % (round(model_score - base_model_score, 3))) + '    distribution  ' + str(
                                    "%+g" % (round(distribution - base_distribution, 3))) + '   Clusters:  ' + str(
                                    best_number_of_clusters))
                            base_model_score = model_score
                            base_distribution = distribution
                            new = new.drop(new_column, axis=1)
                            decider = 1
                            if counter == 30:
                                print(Fore.BLUE + 'Score:  ' + str(model_score) + '    Distribution:  ' + str(
                                    distribution))
                                counter = 0
                        else:
                            print(Fore.GREEN + str(runner) + '. ' + new_column + '     Survived' + '    score  ' + str(
                                "%+g" % (round(model_score - base_model_score, 3))) + '    distribution  ' + str(
                                "%+g" % (round(distribution - base_distribution, 3))) + '   Clusters:  ' + str(
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
    def __init__(self, std, num_of_clusters, method):
        self.std = std
        self.num_of_clusters = num_of_clusters + 1
        self.method = method

    def models(self):
        K = range(2, self.num_of_clusters)
        if self.method == 'KMeans':
            models = [KMeans(n_clusters=k, random_state=42).fit(self.std) for k in K]
        elif self.method == 'GMM':
            models = [GMM(k, random_state=116, covariance_type='full').fit(self.std) for k in K]
        dist = pairwise_distances(self.std)
        np.fill_diagonal(dist, 0)
        calinski = [calinski_harabasz_score(self.std, model.predict(self.std)) for model in models]
        david = [davies_bouldin_score(self.std, model.predict(self.std)) for model in models]
        Dunn = [dunn(dist, model.predict(self.std)) for model in models]
        silh = [silhouette_score(dist, model.predict(self.std), metric='precomputed') for model in models]
        self.david = (np.max(david) / np.array(david)).tolist()
        self.calinski = calinski
        self.dunn = Dunn
        self.silh = silh
        self.models = models

    def metrics(self):
        models_and_metrics.models(self)
        scores = {}
        scores['david'] = self.david
        scores['dunn'] = self.dunn
        scores = pd.DataFrame(scores)
        for column in scores.columns.values:
            if np.any((scores[column].values == float('inf'))):
                scores = scores.drop(column, axis=1)
                continue
        scores['calinski'] = self.calinski
        scores['silh'] = self.silh
        scores = pd.DataFrame(MinMaxScaler().fit_transform(scores), index=np.arange(2, self.num_of_clusters),
                              columns=scores.columns)

        scores['silh'] = 2 * scores['silh']
        try:
            scores['dunn'] = 2 * scores['dunn']
        except:
            pass
        scores['final'] = scores.sum(axis=1)
        pretendents = scores['final'].sort_values(ascending = False).index.values.tolist()[:3]
        for pretendent in pretendents:
            model = self.models[pretendent - 2]
            distribution = pd.Series(model.predict(self.std)).value_counts().std()/pd.Series(model.predict(self.std)).value_counts().mean() * self.num_of_clusters ** (1/10) / pretendent ** (1/10)
            if distribution <= 1:
                best_number_of_clusters = pretendent
                break
            else:
                pass
            if pretendent == pretendents[-1]:
                best_number_of_clusters = pretendents[0]
        best_model = self.models[best_number_of_clusters-2]
        calinski_to_silhouette_scale = MinMaxScaler((np.min(self.silh) - 0.000001,np.max(self.silh))).fit_transform(scores['calinski'].values.reshape(-1,1)).reshape(-1).tolist()
        model_score = silhouette_score(self.std, best_model.predict(self.std)) * 0.7 + 0.3 * calinski_to_silhouette_scale[best_number_of_clusters-2]
        self.pretendents = pretendents
        self.calinski_to_silhouette_scale = calinski_to_silhouette_scale
        return best_number_of_clusters, model_score

    @property
    def get_pretendents_(self):
        return self.pretendents



def xgb_validation(data, labels, pretendents, new_column, algorithm):
    d = pd.DataFrame({'columns': data.columns.values})
    for pretendent in pretendents:
        model = eval(str(algorithm) + '_model')(data, pretendent)
        labels = model.predict(data)
        x = data
        y = labels
        model = XGBClassifier(n_estimators=100, alpha=5, max_depth=10, seed=12345, min_child_weight=3,
                              colsammple_bytree=0.3, learning_rate=0.05, silent=True, verbosity=0)
        model.fit(x,y)
        d = pd.merge(d, pd.DataFrame({pretendent: model.feature_importances_}, index=data.columns.values),
                     left_on='columns',
                     right_index=True)
    d = d.set_index('columns')
    if d.sum(axis = 1)[new_column[0]] <= (1/data.shape[1])/40:
        verdict = 'reject'
    else:
        verdict = 'approve'
    return verdict


def large_set(data, max_num_clusters=20, iterations=10, thresh_1=1, thresh_2=0.6, base=[], Backelim=True):
    best_features, _ = wrap(data, max_num_clusters=max_num_clusters, iterations=iterations,
                            thresh_1=thresh_1, thresh_2=thresh_2, base=base, Backelim=Backelim)
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