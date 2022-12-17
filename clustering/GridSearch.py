from itertools import combinations
from CFS.k_prototypes import models_and_metrics
from clustering import *
from CFS.helpers import C
from tqdm import tqdm
from xgboost import XGBClassifier


def GridSearch(data,combs, models = ['KPrototypes'],max_num_clusters = 5):
    Results = []
    with tqdm(total=len(combs) * len(models)) as progress:
        for model in models:
            results = {}
            for columns in combs:
                competitor_data = data.loc[:, columns]
                if competitor_data.select_dtypes(['int64','int32','int','object']).shape[1] == competitor_data.shape[1]:
                    best_number_of_clusters, model_score = models_and_metrics(competitor_data, max_num_clusters,
                                                                          method = 'KModes').metrics()
                elif competitor_data.select_dtypes(['float64','float32','float']).shape[1] == competitor_data.shape[1]:
                    best_number_of_clusters, model_score = models_and_metrics(competitor_data, max_num_clusters,
                                                                              method='KMeans').metrics()
                else:
                    best_number_of_clusters, model_score = models_and_metrics(competitor_data, max_num_clusters,
                                                                              method='KPrototypes').metrics()
                best_model = eval(model + '_model')(competitor_data, best_number_of_clusters)
                labels = best_model.predict(competitor_data)
                distribution = pd.Series(labels).value_counts().std() / pd.Series(labels).value_counts().mean()
                results[tuple(columns)] = [best_number_of_clusters, model_score, distribution, model]
                progress.update(1)

        Results += [results]
    return Results


def subs(n_jobs, min_subset_size,max_subset_size, initial_set, base = []):
    sizes = np.arange(min_subset_size, max_subset_size)
    all_combs = []
    for size in sizes:
        if base:
            for combination in combinations(initial_set, size):
                if len(list(set(combination) & set(base))) == len(base):
                    all_combs += [list(combination)]
        else:
            for combination in combinations(initial_set, size):
                all_combs += [list(combination)]
    one_core_size = int(round(len(all_combs) / n_jobs,0))
    all_combs = np.array(all_combs)[np.random.permutation(len(all_combs))].tolist()
    cores = []
    for i in range(n_jobs-1):
        core = all_combs[:one_core_size]
        all_combs = all_combs[one_core_size:]
        cores+=[core]
    cores += [all_combs]
    return cores




def pre_grid(data, max_num_clusters, iterations, Model):
    columns = data.columns.values.tolist()
    Sets = []
    for i in range(iterations):
        Sets += [random.sample(columns,np.random.randint(2,len(columns),1)[0])]
    d = pd.DataFrame({'columns': data.columns.values})
    with tqdm(total = (max_num_clusters - 4) * len(Sets)) as progress:
        for Set in Sets:
            for n_clusters in range(5, max_num_clusters+1):
                model = eval(str(Model) + '_model')(data.loc[:,Set], n_clusters)
                labels = model.predict(data.loc[:,Set])
                x = data
                y = labels
                model = XGBClassifier(n_estimators=100, alpha=5, max_depth=10, seed=12345, min_child_weight=3,
                                  colsammple_bytree=0.3, learning_rate=0.05, silent=True, verbosity=0)
                model.fit(x,y)
                d = pd.merge(d, pd.DataFrame({n_clusters: model.feature_importances_}, index=data.columns.values),
                         left_on='columns',
                         right_index=True, how ='outer')
                progress.update(1)
    d = d.set_index('columns')
    return d
