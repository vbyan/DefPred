from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from preprocessing import seperate_dtypes

class FilterMethods:
    def __init__(self, data, corr_thresh=0.9, variance_thresh = 0, base = []):
        self.data = data
        self.corr_thresh = corr_thresh
        self.variance_thresh = variance_thresh
        self.base = base

    def constant_variance(self):
        vs_constant = VarianceThreshold(threshold=self.variance_thresh)
        vs_constant.fit(MinMaxScaler().fit_transform(self.data))
        constant_columns = [column for column in self.data.columns
                            if column not in self.data.columns[vs_constant.get_support()]]
        constant_cat_columns = [column for column in self.data.columns
                                if (self.data[column].dtype == "O" and len(self.data[column].unique()) == 1)]
        all_constant_columns = constant_cat_columns + constant_columns
        cleared_data = self.data.drop(labels=all_constant_columns, axis=1)
        return cleared_data

    def correlation(self):
        self.constant_cleared_data = FilterMethods.constant_variance(self)
        numerical_features, categorical_features = seperate_dtypes(self.constant_cleared_data)
        corr_features = set()
        if numerical_features:
            numerical = self.constant_cleared_data.loc[:,numerical_features]
            k = numerical.columns.values
            if self.base:
                for i in self.base:
                    if np.where(k == i)[0].tolist():
                        k = np.delete(k, np.where(k == i))
                        k = np.concatenate(([i], k))
            corr_matrix = numerical.loc[:, k].corr(method = 'pearson')
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) >= self.corr_thresh:
                        colname = corr_matrix.columns[i]
                        corr_features.add(colname)
        if categorical_features:
            categorical = self.constant_cleared_data.loc[:,categorical_features]
            k = categorical.columns.values
            if self.base:
                for i in self.base:
                    if np.where(k == i)[0].tolist():
                        k = np.delete(k, np.where(k == i))
                        k = np.concatenate(([i], k))
            corr_matrix = categorical.loc[:, k].astype(int).corr(method = 'spearman')
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) >= self.corr_thresh:
                        colname = corr_matrix.columns[i]
                        corr_features.add(colname)

        constant_and_correlation_cleared_data = self.constant_cleared_data.drop(labels=corr_features, axis=1)
        return constant_and_correlation_cleared_data

    def filter(self):
        self.constant_and_correlation_cleared_data = FilterMethods.correlation(self)
        return self.constant_and_correlation_cleared_data
