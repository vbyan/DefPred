import numpy as np
import pandas as pd
import pickle


def data_preprocessing(raw_data):
    def date_parts(x ,part):
        try:
            if part == 'month':
                return x.strftime('%m')
            if part == 'day':
                return x.strftime('%d')
            if part == 'year':
                return x.strftime('%Y')
        except:
            return np.nan
    preprocessed_data = raw_data.copy()
    preprocessed_data = preprocessed_data.drop(['FNOTE3'], axis=1)
    folder = '/content/gdrive/MyDrive/Loan_Scoring_Third_Phase/AraratBank/Preprocessing/'

    with open(folder + 'cat_features.txt' ,'rb') as file:
        cat_features = pickle.load(file)

    with open(folder + 'label_encoders.txt' ,'rb') as file:
        label_encoders = pickle.load(file)
    for ft in cat_features:
        preprocessed_data[ft] = label_encoders[ft].transform(preprocessed_data[ft])



    with open(folder + 'nan_features.txt' ,'rb') as file:
        nan_features = pickle.load(file)

    with open(folder + 'overdue_rel_features.txt' ,'rb') as file:
        overdue_related_features = pickle.load(file)

    with open(folder + 'payment_rel_features.txt' ,'rb') as file:
        payment_related_features = pickle.load(file)

    preprocessed_data = preprocessed_data.drop(overdue_related_features, axis = 1)
    preprocessed_data = preprocessed_data.drop(payment_related_features, axis = 1)

    date_features = preprocessed_data.select_dtypes('datetime').columns.values.tolist()
    for ft in date_features:
        preprocessed_data[ft + '_day'] = preprocessed_data[ft].apply(lambda x :date_parts(x ,'day'))
        preprocessed_data[ft + '_month'] = preprocessed_data[ft].apply( lambda x :date_parts(x ,'month'))
        preprocessed_data[ft + '_year'] = preprocessed_data[ft].apply( lambda x :date_parts(x ,'year'))
        preprocessed_data[ft] = preprocessed_data[ft].astype(str).replace('NaT', np.nan)
        preprocessed_data[ft] = preprocessed_data[ft].str.replace('-' ,'').astype('float64')
    preprocessed_data = preprocessed_data.drop(nan_features, axis = 1).dropna()

    return preprocessed_data





def get_data(x, y, loan_type, train_test, y_defaulttime = None):
    """
    available loan_type values
    - gold
    - card
    - sme
    - large
    - consumer_unsecured
    - consumer_secured
    - mortgage
    - agro
    """

    folder = '/content/gdrive/MyDrive/Loan_Scoring_Third_Phase/AraratBank/Preprocessing/'
    indices = np.load(
        folder + 'Types/' + loan_type + '_' + train_test + '.npy')

    data = x.copy()
    data['y'] = y
    if y_defaulttime is not None:
        data['defaulttime'] = y_defaulttime
    data = data.loc[indices]
    if y_defaulttime is not None:
        new_x = data.drop(['y','defaulttime'], axis=1)
    else:
        new_x = data.drop('y', axis=1)
    new_y = data['y'].values

    if y_defaulttime is not None:
        new_defaulttime = data['defaulttime']
        return new_x, new_y, new_defaulttime
    else:
        return new_x, new_y
