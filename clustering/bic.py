import pandas as pd
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq
from sklearn.linear_model import LinearRegression
from kneed import KneeLocator

def local_min(bic):
    global diffs
    global args
    diffs = []
    args = []
    for i in range(1, len(bic) - 1):
        if (bic[i - 1] > bic[i]) & (bic[i] < bic[i + 1]):
            diffs += [abs(bic[i - 1] - bic[i]) + abs(bic[i] - bic[i + 1])]
            args += [i + 1]
    k = -1
    stop = np.array(['run'])
    if len(diffs) > 0:
        while stop:
            if (diffs[k] / 2 >= (np.max(bic) - np.min(bic)) * 0.1) & (bic[args[k] - 2] == np.min(bic[:args[-1] - 1])):
                best_number_of_clusters = args[k]
                return best_number_of_clusters
                break
            else:
                k = k - 1
            if abs(k) > len(diffs):
                stop = np.delete(stop, np.where(stop == 'run'))
                best_number_of_clusters = 'Continue'
    else:
        best_number_of_clusters = 'Continue'
    return best_number_of_clusters

def angle(bic, min):
    K = range(1, len(bic) + 1)
    right_side_K = np.array(K[min - 1:]).reshape(-1, 1)
    left_side_K = np.array(K[:min]).reshape(-1, 1)
    right_side_bic = np.array(bic[min - 1:]).reshape(-1, 1)
    left_side_bic = np.array(bic[:min]).reshape(-1, 1)

    model_right = LinearRegression()
    model_right.fit(np.array(right_side_K).reshape(-1, 1), np.array(right_side_bic).reshape(-1, 1))
    model_left = LinearRegression()
    model_left.fit(np.array(left_side_K).reshape(-1, 1), np.array(left_side_bic).reshape(-1, 1))

    l = model_left.predict(np.array([min]).reshape(-1, 1))
    r = model_right.predict(np.array([min]).reshape(-1, 1))

    left_bic = model_left.predict(left_side_K)
    right_bic = model_right.predict(right_side_K) - r[0, 0] + l[0, 0]
    smooth_bic = np.append(left_bic, right_bic[1:, :])

    x_coords_1 = [min, min - 1]
    y_coords_1 = [smooth_bic[min - 1], smooth_bic[min - 2]]
    x_coords_2 = [min + 1, min]
    y_coords_2 = [smooth_bic[min], smooth_bic[min - 1]]

    A1 = vstack([x_coords_1, ones(len(x_coords_1))]).T
    m1, c = lstsq(A1, y_coords_1)[0]
    A2 = vstack([x_coords_2, ones(len(x_coords_2))]).T
    m2, c = lstsq(A1, y_coords_2)[0]
    return m1, m2

def best_number(K, bic):
    kneelocator = KneeLocator(range(K[-1], K[0] -1, -1), bic, curve='convex', direction='increasing', S=1)
    best_number_of_clusters = K[-1] + K[0] - kneelocator.knee
    return best_number_of_clusters