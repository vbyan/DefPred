from preprocessing import hsbc_preprocessing
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from helpers import parser_with_nan, merge_overlaps
from dateutil import parser
from sklearn.model_selection import KFold
from helpers import best_threshold
from joblib import Parallel,delayed
from collections.abc import Iterable
from sklearn.base import clone
import os

if os.path.exists('/content/gdrive/MyDrive/Loan_Scoring_Third_Phase/HSBC/'):
    folder = '/content/gdrive/MyDrive/Loan_Scoring_Third_Phase/HSBC/'
else:
    folder = 'G:\My Drive\Loan_Scoring_Third_Phase\HSBC/'

from fuzzywuzzy import fuzz


def correct_transform(new_data):
    def similar_strings(string1, string2):
        if fuzz.partial_ratio(string1.lower(), string2.lower()) == 100:
            return True
        else:
            if fuzz.ratio(string1.lower(), string2.lower()) >= 70:
                return True
            else:
                return False

    with open(folder + 'Preprocessing/raw_columns.txt', 'rb') as file:
        raw_columns = pickle.load(file)
    with open(folder + 'Preprocessing/raw_dtypes.txt', 'rb') as file:
        dtypes = pickle.load(file)

    new_data_columns = new_data.columns.values.tolist()
    transformed_data = pd.DataFrame([])
    for i in raw_columns:
        if i in new_data_columns:
            transformed_data[i] = new_data[i]
        else:
            for j in new_data_columns:
                if similar_strings(i, j):
                    transformed_data[i] = new_data[j]
        try:
            transformed_data[i] = transformed_data[i].astype(dtypes[i])
        except:
            pass
    return transformed_data

def load_params(database, name):
    with open(folder + 'Preprocessing/merge/' + database + '/' + name + '.txt', 'rb') as file:
        params = pickle.load(file)
    return params

def save_params(database,name):
    with open(folder + 'Preprocessing/merge/' + database + '/' + name + '.txt', 'wb') as file:
        pickle.dump(eval(name), file)



crrc_mean_1 = load_params('crrc', 'mean_1')
crrc_mean_2 = load_params('crrc', 'mean_2')
crrc_mean_3 = load_params('crrc', 'mean_3')
crrc_mean_4 = load_params('crrc', 'mean_4')
crrc_mean_5 = load_params('crrc', 'mean_5')
crrc_drop_1 = load_params('crrc', 'drop_1')
crrc_drop_4 = load_params('crrc', 'drop_4')
crrc_drop_5 = load_params('crrc', 'drop_5')
crrc_max_1 = load_params('crrc', 'max_1')
crrc_labelencoders_1 = load_params('crrc', 'labelencoders_1')

wvs_mean_1 = load_params('wvs', 'mean_1')
wvs_mean_2 = load_params('wvs', 'mean_2')
wvs_mean_3 = load_params('wvs', 'mean_3')
wvs_mean_4 = load_params('wvs', 'mean_4')
wvs_mean_5 = load_params('wvs', 'mean_5')
wvs_drop_1 = load_params('wvs', 'drop_1')
wvs_drop_4 = load_params('wvs', 'drop_4')
wvs_drop_5 = load_params('wvs', 'drop_5')
wvs_bins = load_params('wvs', 'bins')
wvs_labelencoders = load_params('wvs', 'labelencoders')

class merge_prepare:
    def __init__(self, hsbc, hsbc_preprocessed, database):
        self.hsbc = hsbc
        self.hsbc_preprocessed = hsbc_preprocessed
        self.database = database

        """crrc_mean_1 = load_params('crrc','mean_1')
        crrc_mean_2 = load_params('crrc', 'mean_2')
        crrc_mean_3 = load_params('crrc', 'mean_3')
        crrc_mean_4 = load_params('crrc', 'mean_4')
        crrc_mean_5 = load_params('crrc', 'mean_5')
        crrc_drop_1 = load_params('crrc','drop_1')
        crrc_drop_4 = load_params('crrc','drop_4')
        crrc_drop_5 = load_params('crrc','drop_5')
        crrc_max_1 = load_params('crrc','max_1')
        crrc_labelencoders_1 = load_params('crrc','labelencoders_1')

        wvs_mean_1 = load_params('wvs', 'mean_1')
        wvs_mean_2 = load_params('wvs', 'mean_2')
        wvs_mean_3 = load_params('wvs', 'mean_3')
        wvs_mean_4 = load_params('wvs', 'mean_4')
        wvs_mean_5 = load_params('wvs', 'mean_5')
        wvs_drop_1 = load_params('wvs', 'drop_1')
        wvs_drop_4 = load_params('wvs', 'drop_4')
        wvs_drop_5 = load_params('wvs', 'drop_5')
        wvs_t = load_params('wvs', 't')
        wvs_labelencoders = load_params('wvs', 'labelencoders')

        self.crrc_mean_1 = crrc_mean_1
        self.crrc_mean_2 = crrc_mean_2
        self.crrc_mean_3 = crrc_mean_3
        self.crrc_mean_4 = crrc_mean_4
        self.crrc_mean_5 = crrc_mean_5
        self.crrc_drop_1 = crrc_drop_1
        self.crrc_drop_4 = crrc_drop_4
        self.crrc_drop_5 = crrc_drop_5
        self.crrc_max_1 = crrc_max_1
        self.crrc_labelencoders_1 = crrc_labelencoders_1

        self.wvs_mean_1 = wvs_mean_1
        self.wvs_mean_2 = wvs_mean_2
        self.wvs_mean_3 = wvs_mean_3
        self.wvs_mean_4 = wvs_mean_4
        self.wvs_mean_5 = wvs_mean_5
        self.wvs_drop_1 = wvs_drop_1
        self.wvs_drop_4 = wvs_drop_4
        self.wvs_drop_5 = wvs_drop_5
        self.wvs_t = wvs_t
        self.wvs_labelencoders = wvs_labelencoders"""


    def hsbc_overlaps(self):
        hsbc = self.hsbc.copy()
        hsbc_preprocessed = self.hsbc_preprocessed.copy()
        if self.database == 'wvs':
            hsbc['Total income'] = hsbc_preprocessed['Total income']
            hsbc['Approved\nlimit'] = hsbc['Approved\nlimit'].fillna(wvs_mean_1)

            hsbc_overlapping_features = ['Citizenship', 'Marital Status', '# of children', 'Education Level',
                                         'Home Ownership status', 'Occupation type', 'Industry', 'Customer segment',
                                         'Total income']
            hsbc_overlaps = hsbc.loc[:, hsbc_overlapping_features]
            hsbc_overlaps['Citizenship'] = np.where(hsbc_overlaps['Citizenship'] == 'AM', 1, 2)
            hsbc_overlaps['Marital Status'] = hsbc_overlaps['Marital Status'].replace({'MARRIED': 1,
                                                                                       'SINGLE': 6,
                                                                                       'DIVORCED/SEPARATED': 3,
                                                                                       'WIDOWED': 5,
                                                                                       'LIVINGTOGETHER': 2})
            hsbc_overlaps['Education Level'] = hsbc_overlaps['Education Level'].replace({'GRADUATE': 4, 'SECONDARY': 2,
                                                                                         'TERTIARY': 3,
                                                                                         'POSTGRADUATE': 4,
                                                                                         'PRIMARY': 1, 'NONE': 1})
            hsbc_overlaps['Home Ownership status'] = hsbc_overlaps['Home Ownership status'].replace(
                {'BELONGS TO PARENTS': 2,
                 'OWN': 1, 'BELONGS TO RELATIVES': 3, 'RENTING': 1, 'BELONGS TO SPOUSE': 1, 'OTHER': 1})
            hsbc_overlaps['Occupation type'] = hsbc_overlaps['Occupation type'].replace(
                {'Specialist/Professional roles': 1,
                 'Unskilled/Non specialist': 8, 'Management roles': 2, 'Skilled manuals/craftsmen': 6, 'Unemployed': 0,
                 'Service roles': 5, 'skilled manuals/craftsmen': 6})
            hsbc_overlaps['Industry'] = hsbc_overlaps['Industry'].replace(
                {'Information and communication / IT': 2, 'Manufacturing': 2,
                 'Retail Trade': 2, 'Retail trade': 2, 'Import': 2, 'Customer Service': 2,
                 'Health and Population social services': 1,
                 'Charity, human rights protection, peace and security missions': 3,
                 'Public administration, Administration and support activity': 1,
                 'Protection / Military': 1, 'Education': 1, 'Other': 2, 'Financial, audit and insurance activity': 2
                    , 'Professional, scientific and technical activity': 2, 'Construction and architecture': 2,
                 'EMBASSY/CONSULATE': 1,
                 'Electricity, gas supply,Water supply, sewerage, waste-management and processing': 1,
                 'Culture, entertainment, Sport': 2, 'Wholesale trade': 2, 'Transportation and storage economy': 2,
                 'Mining industry, Exploitation of open pits, export': 2,
                 'Justice': 1, 'Residential and Public Food organizing': 1, 'Agriculture, Forest economy, Fishing': 2,
                 'Real estate related activity': 2,
                 'Religion': 3, 'Export': 2, 'financial, audit and insurance activity': 2})
            hsbc_overlaps['Customer segment'] = hsbc_overlaps['Customer segment'].replace(
                {'MASS': 4, 'PLUS': 2, 'STATUS': 1})
            t = hsbc_overlaps['Total income'].copy()
            t = pd.cut(t,wvs_bins, labels = False, right = True, include_lowest = True)
            t = t.fillna(11).astype('int64')
            hsbc_overlaps['Total income'] = t + 1

            cat_ft = ['Citizenship', 'Marital Status', 'Home Ownership status', 'Occupation type', 'Industry']
            num_ft = ['# of children', 'Education Level', 'Customer segment', 'Total income']
            for i, mean_value in zip(cat_ft,wvs_mean_2):
                hsbc_overlaps[i] = hsbc_overlaps[i].fillna(mean_value)
                hsbc_overlaps[i] = hsbc_overlaps[i].astype('int64')
            for i, mean_value in zip(num_ft,wvs_mean_3):
                hsbc_overlaps[i] = hsbc_overlaps[i].fillna(mean_value)
                hsbc_overlaps[i] = hsbc_overlaps[i].astype('float64')

            return hsbc_overlaps

        elif self.database == 'crrc':
            hsbc['Employed since (current employer)'] = hsbc['Employed since (current employer)'].apply(
                lambda x: parser_with_nan(x))
            hsbc['Applic\ndate'] = hsbc['Applic\ndate'].apply(lambda x: parser.parse(x))
            hsbc['Settlement_type'] = np.where(hsbc['Residential Address'].isna(), np.nan,
                                               np.where(hsbc['Residential Address'].str.contains('VILLAGE'), 'RURAL',
                                                        np.where(hsbc['Residential Address'].str.contains('YEREVAN'),
                                                                 'CAPITAL', 'URBAN')))
            hsbc['Has_other_income'] = np.where(hsbc['Other income (in AMD)'].isna(), 0,
                                                np.where(hsbc['Other income (in AMD)'] == 0, 0, 1))

            hsbc_overlaps = hsbc.loc[:, ['Employed since (current employer)', 'Industry', 'Salary (in AMD)',
                                         'Debit turnover for last 6 months (customer level)', 'Citizenship',
                                         'Education Level',
                                         '# of children', 'Marital Status', 'Settlement_type', 'Gender',
                                         'Car Ownership Status Yes/No',
                                         'Has_other_income']]
            hsbc_overlaps['Debit turnover for last 6 months (customer level)'] = np.where(
                ~hsbc_overlaps['Debit turnover for last 6 months (customer level)'].isna(), 1, 0)
            hsbc_overlaps['Citizenship'] = hsbc_overlaps['Citizenship'].replace(
                {'AM': 'Armenian', 'RU': 'Russian', 'GE': 'Georgian',
                 'AZ': 'Azerbaijani'})

            hsbc_overlaps['Employed since (current employer)'] = (hsbc['Applic\ndate'] - hsbc[
                'Employed since (current employer)']) / pd.Timedelta('365 days')
            hsbc_overlaps['Citizenship'] = np.where(hsbc_overlaps['Citizenship'].str.len() == 2, 'Other',
                                                    hsbc_overlaps['Citizenship'])
            hsbc_overlaps['Salary (in AMD)'] = hsbc_preprocessed['Total income'] / 12

            hsbc_overlaps['Industry'] = hsbc_overlaps['Industry'].replace(
                {'Information and communication / IT': 3, 'Manufacturing': 6, 'Retail Trade': 4, 'Retail trade': 4,
                 'Import': 4, 'Customer Service': 10,
                 'Health and Population social services': 8,
                 'Charity, human rights protection, peace and security missions': 14,
                 'Public administration, Administration and support activity': 5, 'Protection / Military': 5,
                 'Education': 2, 'Other': 3,
                 'Financial, audit and insurance activity': 12, 'Professional, scientific and technical activity': 3,
                 'Construction and architecture': 7,
                 'EMBASSY/CONSULATE': 5,
                 'Electricity, gas supply,Water supply, sewerage, waste-management and processing': 11,
                 'Culture, entertainment, Sport': 3,
                 'Wholesale trade': 4, 'Transportation and storage economy': 9,
                 'Mining industry, Exploitation of open pits, export': 15, 'Justice': 5,
                 'Residential and Public Food organizing': 8,
                 'Agriculture, Forest economy, Fishing': 1, 'Real estate related activity': 16, 'Religion': 3,
                 'Export': 4, 'financial, audit and insurance activity': 12})
            hsbc_overlaps['Citizenship'] = hsbc_overlaps['Citizenship'].replace(
                {'Armenian': 0, 'Russian': 2, 'Other': 4, 'Georgian': 3, 'Azerbaijani': 5})

            hsbc_overlaps['Education Level'] = hsbc_overlaps['Education Level'].replace(
                {'GRADUATE': 2, 'SECONDARY': 3, 'TERTIARY': 0, 'POSTGRADUATE': 6, 'PRIMARY': 5, 'NONE': 7})
            hsbc_overlaps['Marital Status'] = hsbc_overlaps['Marital Status'].replace(
                {'MARRIED': 1, 'SINGLE': 3, 'DIVORCED/SEPARATED': 5, 'WIDOWED': 2, 'LIVINGTOGETHER': 4})
            hsbc_overlaps['Settlement_type'] = hsbc_overlaps['Settlement_type'].replace(
                {'CAPITAL': 0, 'RURAL': 1, 'URBAN': 2, 'nan': 0})
            hsbc_overlaps['Gender'] = hsbc_overlaps['Gender'].replace({'F': 0, 'M': 1})
            hsbc_overlaps['Car Ownership Status Yes/No'] = hsbc_overlaps['Car Ownership Status Yes/No'].replace(
                {'N': 0, 'Y': 1, ' ': 2})

            cat_ft = hsbc_overlaps.drop(
                ['Employed since (current employer)', 'Salary (in AMD)', '# of children', 'Education Level'],
                axis=1).columns.values.tolist()
            num_ft = ['Employed since (current employer)', 'Salary (in AMD)', '# of children', 'Education Level']
            for i, mean_value in zip(cat_ft, crrc_mean_1):
                hsbc_overlaps[i] = hsbc_overlaps[i].fillna(mean_value)
                hsbc_overlaps[i] = hsbc_overlaps[i].astype('int64')
            for i, mean_value in zip(num_ft,crrc_mean_2):
                hsbc_overlaps[i] = hsbc_overlaps[i].fillna(mean_value)
                hsbc_overlaps[i] = hsbc_overlaps[i].astype('float64')

            return hsbc_overlaps

    def database_overlaps(self):
        if self.database == 'wvs':
            wvs = pd.read_csv(folder + 'Data/main/wvs_raw.csv')
            overlaps = wvs.loc[:, ['Q269', 'Q273', 'Q274', 'Q275R', 'Q271', 'Q281', 'Q284', 'Q287',
                                   'Q288R']].dropna()

            for i in ['Q269', 'Q273', 'Q271', 'Q281', 'Q284']:
                overlaps[i] = overlaps[i].astype('int64')
            overlaps = overlaps.sample(32000, random_state=42)
            self.wvs_overlaps = overlaps

            return overlaps

        elif self.database == 'crrc':
            crrc = pd.read_spss(folder + 'Data/main/crrc_raw.sav')
            relatives = crrc.loc[:,
                        pd.Series(crrc.columns.values)[(pd.Series(crrc.columns.values).str.contains('HHM')) &
                                                       (pd.Series(crrc.columns.values).str.contains(
                                                           'RELR'))].values.tolist()]
            other_income = crrc.loc[:, pd.Series(crrc.columns.values)[
                                           pd.Series(crrc.columns.values).str.contains('INCSOU')].values.tolist()]
            other_income = other_income.drop(['INCSOUCO', 'INCSOUAB', 'INCSOUAG', 'INCSOUSL'], axis=1)
            crrc['Children'] = relatives[relatives == 'Son / daughter'].replace('Son / daughter', 1).sum(axis=1,
                                                                                                         skipna=True)
            crrc['Has_other_income'] = np.where(
                other_income[other_income == 'Yes'].replace('Yes', 1).sum(axis=1, skipna=True) > 0, 1, 0)

            overlaps = crrc.loc[:, ['WORKYRS', 'WORKSEC', 'PERSINC', 'BANKACC', 'ETHNIC', 'RESPEDU', 'Children',
                                    'RESPMAR', 'STRATUM', 'RESPSEX', 'OWNCARS', 'Has_other_income']]
            overlaps.replace({"Don't know": np.nan, 'Refuse to answer': np.nan, 'Interviewer error': np.nan},
                             inplace=True)

            overlaps['WORKYRS'] = overlaps['WORKYRS'].replace('Legal skip', 2020)
            overlaps['WORKYRS'] = overlaps['WORKYRS'].astype('float64').apply(lambda x: 2020 - x)
            overlaps['WORKSEC'] = overlaps['WORKSEC'].replace('Legal skip', 'Unemployed')
            overlaps['PERSINC'] = overlaps['PERSINC'].astype(str).replace('0', 'USD 0').str.split('USD ').str[1]
            overlaps['PERSINC'] = np.where(overlaps['PERSINC'].str.contains('-'),
                                           (overlaps['PERSINC'].str.split('-').str[0].astype('float64') +
                                            overlaps['PERSINC'].str.split('-').str[1].astype('float64')) / 2 * 480
                                           , overlaps['PERSINC'])
            overlaps['PERSINC'] = overlaps['PERSINC'].replace({'50': 50 * 480, '0': 0})

            overlaps['WORKSEC'] = overlaps['WORKSEC'].replace({'Unemployed': 0, 'Agriculture, hunting, and forestry': 1,
                                                               'Education': 2, 'Other': 3,
                                                               'Wholesale / retail trade, repair, personal and household goods': 4,
                                                               'Government, public administration and defense': 5,
                                                               'Manufacturing': 6, 'Construction': 7,
                                                               'Healthcare and social work': 8,
                                                               'Transport, storage, communication': 9,
                                                               'Hotels, restaurants, or cafes': 10,
                                                               'Electricity, gas, or water supply': 11,
                                                               'Financial intermediation / Banking': 12,
                                                               'Mass media': 13, 'Civil society / NGOs': 14,
                                                               'Mining and quarrying': 15,
                                                               'Real estate, property- and rent-related activities': 16})
            overlaps['ETHNIC'] = overlaps['ETHNIC'].replace(
                {'Armenian': 0, 'Kurd or Yezid': 1, 'Russian': 2, 'Georgian': 3, 'Other': 4, 'Azerbaijani': 5,
                 'Other Caucasian': 6})
            overlaps['RESPEDU'] = overlaps['RESPEDU'].replace(
                {'Competed secondary education': 0, 'Secondary technical education': 1, 'Completed higher education': 2,
                 'Incomplete secondary education': 3, 'Incomplete higher education': 4, 'Primary education': 5,
                 'Post-graduate degree': 6,
                 'No primary education': 7}).astype('float64')
            overlaps['RESPMAR'] = overlaps['RESPMAR'].replace(
                {'Married: official state marriage only': 0, 'Married: both religious ceremony and state marriage': 1,
                 'Widow / widower': 2, 'Never married.': 3, 'Cohabiting without civil or religious marriage': 4,
                 'Divorced': 5,
                 'Married: religious ceremony only': 6, 'Separated': 7})
            overlaps['STRATUM'] = overlaps['STRATUM'].replace({'Capital': 0, 'Rural': 1, 'Urban': 2})
            overlaps['RESPSEX'] = overlaps['RESPSEX'].replace({'Female': 0, 'Male': 1})
            overlaps['OWNCARS'] = overlaps['OWNCARS'].replace({'No': 0, 'Yes': 1})
            overlaps['BANKACC'] = overlaps['BANKACC'].replace({'Yes': 1, 'No': 0})
            for i, mean_value in zip(['WORKSEC', 'BANKACC', 'ETHNIC', 'RESPMAR', 'STRATUM', 'RESPSEX', 'OWNCARS', 'Has_other_income'], crrc_mean_3):
                overlaps[i] = overlaps[i].fillna(mean_value)
                overlaps[i] = overlaps[i].astype('int64')

            return overlaps

    def hsbc_whole(self):
        return self.hsbc_preprocessed

    def database_whole(self):
        if self.database == 'wvs':
            wvs = pd.read_csv(folder + 'Data/main/wvs_raw.csv')
            wvs = wvs.loc[self.wvs_overlaps.index.values.tolist(), :]

            return wvs
        elif self.database == 'crrc':
            crrc = pd.read_spss(folder + 'Data/main/crrc_raw.sav')
            relatives = crrc.loc[:,
                        pd.Series(crrc.columns.values)[(pd.Series(crrc.columns.values).str.contains('HHM')) &
                                                       (pd.Series(crrc.columns.values).str.contains(
                                                           'RELR'))].values.tolist()]
            other_income = crrc.loc[:, pd.Series(crrc.columns.values)[
                                           pd.Series(crrc.columns.values).str.contains('INCSOU')].values.tolist()]
            other_income = other_income.drop(['INCSOUCO', 'INCSOUAB', 'INCSOUAG', 'INCSOUSL'], axis=1)
            crrc['Children'] = relatives[relatives == 'Son / daughter'].replace('Son / daughter', 1).sum(axis=1,
                                                                                                         skipna=True)
            crrc['Has_other_income'] = np.where(
                other_income[other_income == 'Yes'].replace('Yes', 1).sum(axis=1, skipna=True) > 0, 1, 0)
            relevant_cat_cols = ['ACTCHORE', 'ACTRESDT', 'ACTCLEAN', 'ACTTHEA', 'ACTREST',
                                 'ACTVLNT', 'ACTCOMM', 'ACTDNCH', 'ACTNRCHN', 'ACTPBLM', 'ACTSPET',
                                 'ACTDON', 'HLTHRAT', 'CLRLABR', 'CLFRDAB', 'GENBREA', 'GENBRER', 'APTINHERT',
                                 'POLDIRN', 'IMPISS1', 'IMPISS2',
                                 'REVEXP1', 'REVEXP2', 'FREESPK', 'EMPLSIT', 'GETJOBF', 'NEWHHJOB', 'JOBLOST',
                                 'RFAEDUC', 'RMOEDUC', 'FLMANDSC',
                                 'EMIGRAT', 'MIGSHRT', 'OWNCOTV',
                                 'OWNDIGC', 'OWNWASH', 'OWNFRDG',
                                 'OWNAIRC', 'OWNCARS', 'OWNLNDP', 'OWNCELL',
                                 'OWNPCELL', 'OWNCOMP', 'SAVINGS', 'DEBTSHH', ]
            relevant_ord_cols = ['TRUBANK', 'GALLTRU', 'DISCPRPR', 'FATEINLF', 'TRUEDUC', 'TRUCRTS', 'TRUARMY',
                                 'TRUPARL', 'TRUEXEC',
                                 'TRUPRES', 'TRUPOLI', 'TRUSTEU', 'QUALINF', 'FAIRTRT', 'JOBSARFN', 'FRQINTR',
                                 'RELSERV', 'ECONSTN',
                                 'NOMONFD', 'MONYTOT', 'SPENDMO', 'FOODDBT', 'UTILDBT', 'CURRUNG', 'FUTRUNG', 'RELCOND',
                                 'MININCR']
            relevant_num_cols = ['HHSIZE', 'HHASIZE', 'IDEALNCH', 'NUMCOTV', 'NUMFRDG', 'NUMDIGC', 'NUMWASH', 'NUMAIRC',
                                 'NUMCELL',
                                 'NUMCARS', 'NUMPCELL', 'NUMCOMP']

            crrc = crrc.loc[:, relevant_cat_cols + relevant_ord_cols + relevant_num_cols]
            crrc = crrc.replace('Legal skip', np.nan)

            trust = ['TRUBANK', 'TRUEDUC', 'TRUCRTS', 'TRUARMY', 'TRUPARL', 'TRUEXEC', 'TRUPRES', 'TRUPOLI', 'TRUSTEU']
            crrc.loc[:, trust] = crrc.loc[:, trust].replace(
                {'Fully distrust': 0, 'Rather distrust': 1, 'Neither trust nor distrust': 2,
                 'Rather trust': 3, 'Fully trust': 4, "Don't know": 2, 'Refuse to answer': 2})
            crrc['GALLTRU'] = crrc['GALLTRU'].replace(
                {'You cannot be too careful': 1, 'Most people can be trusted': 10, "Don't know": 5, 'Break off': 5})
            crrc['DISCPRPR'] = crrc['DISCPRPR'].replace(
                {'Never': 1, 'Always': 10, "Don't know": 5, 'Refuse to answer': 5})
            crrc['FATEINLF'] = crrc['FATEINLF'].replace(
                {'People shape their fate themselves': 10, 'Everything in life is determined by fate': 1,
                 "Don't know": 5, 'Refuse to answer': 5})
            crrc['QUALINF'] = crrc['QUALINF'].replace(
                {'Very poorly': 0, 'Quite poorly': 1, 'In the middle': 2, 'Quite well': 3, 'Very well': 4,
                 "Don't know": 2})
            crrc['FAIRTRT'] = crrc['FAIRTRT'].replace(
                {'Completely disagree': 0, 'Somewhat disagree': 1, 'Somewhat agree': 3, 'Completely agree': 4,
                 "Don't know": 2, 'Refuse to answer': 2})
            crrc['JOBSARFN'] = crrc['JOBSARFN'].replace(
                {'Very dissatisfied': 0, 'Somewhat dissatisfied': 1, 'Average satisfaction': 2,
                 'Somewhat satisfied': 3, 'Very satisfied': 4})

            crrc['FRQINTR'] = crrc['FRQINTR'].replace(
                {'Do not know what the internet is': 0, 'Never': 1, 'Less often': 2, 'At least once a month': 3,
                 'At least once a week': 4,
                 'Every day': 5, "Don't know": 3})
            crrc['RELSERV'] = crrc['RELSERV'].replace(
                {'Never': 0, 'Less often': 1, 'Only on special religious holidays': 2, 'At least once a month': 3,
                 'Once a week': 4, 'More than once a week': 5, 'Every day': 6, "Don't know": 2})
            crrc['ECONSTN'] = crrc['ECONSTN'].replace(
                {'Money is not enough for food': 0, 'Money is enough for food only, but not for clothes': 1,
                 'Money is enough for food and clothes, but not enough for expensive durables': 2,
                 'We can afford to buy some expensive durables': 3,
                 'We can afford to buy anything we need': 4, 'Refuse to answer': 2})
            crrc['NOMONFD'] = crrc['NOMONFD'].replace(
                {'Every day': 0, 'Every week': 1, 'Every month': 2, 'Less often': 3, 'Never': 4,
                 'Refuse to answer': 2, "Don't know": 2, 'Break off': 2})
            money = ['MONYTOT', 'SPENDMO', 'MININCR']
            crrc.loc[:, money] = crrc.loc[:, money].replace(
                {'0': 0, 'Up to USD 50': 1, 'USD 51 - 100': 2, 'USD 101 - 250': 3, 'USD 251 - 400': 4,
                 'USD 401 - 800': 5, 'USD 801 - 1200': 6, 'USD 1200 - 2000': 7, 'More than USD 2000': 8,
                 "Don't know": 4, 'Refuse to answer': 4, 'Break off': 4})
            crrc['FOODDBT'] = crrc['FOODDBT'].replace(
                {'Each week': 0, 'Each month': 1, 'Every other month': 2, 'Less frequently': 3,
                 'Never': 4, 'Refuse to answer': 2, "Don't know": 2})
            crrc['UTILDBT'] = crrc['UTILDBT'].replace(
                {'Each week': 0, 'Each month': 1, 'Every other month': 2, 'Less frequently': 3,
                 'Never': 4, 'Refuse to answer': 2, "Don't know": 2})
            crrc['CURRUNG'] = crrc['CURRUNG'].replace(
                {'Lowest': 1, 'Highest': 10, "Don't know": 5, 'Refuse to answer': 5})
            crrc['FUTRUNG'] = crrc['FUTRUNG'].replace(
                {'Lowest': 1, 'Highest': 10, "Don't know": 5, 'Refuse to answer': 5})
            crrc['RELCOND'] = crrc['RELCOND'].replace(
                {'Very poor': 0, 'Poor': 1, 'Fair': 2, 'Good': 3, 'Very good': 4, "Don't know": 2,
                 'Refuse to answer': 2})
            for col in relevant_cat_cols:
                labelencoder = crrc_labelencoders_1[col]
                crrc[col] = labelencoder.transform(crrc[col].astype(str).astype(str))

            crrc['IDEALNCH'] = crrc['IDEALNCH'].replace({'Whatever number the God will give us': 2, "Don't know ": 2})
            crrc.loc[:, relevant_num_cols] = crrc.loc[:, relevant_num_cols].replace(
                {'Refuse to answer': 1, "Don't know": 1})
            for col in relevant_cat_cols + relevant_ord_cols + relevant_num_cols:
                crrc[col] = crrc[col].astype('float64')
            for col in relevant_cat_cols:
                crrc[col] = crrc[col].astype('int64')

            return crrc


def postprocess(merged_data, database):
    if database == 'wvs':
        def leave_only_questions(wvs):
            s = pd.Series(wvs.columns.values)
            new_columns = s[s.str.startswith('Q')].values.tolist()
            return wvs.loc[:, new_columns]

        hsbc_part = merged_data.loc[:, :'Debt/Income']
        wvs_part = merged_data.loc[:, 'version':]
        with open(folder + 'Preprocessing/wvs_columns.txt', 'rb') as file:
            wvs_columns = pickle.load(file)
        wvs_part_cat = wvs_part.loc[:, wvs_columns['categorical']]
        wvs_part_ord = wvs_part.loc[:, wvs_columns['ordinal']]
        wvs_part_num = wvs_part.loc[:, wvs_columns['numerical']]
        for i in wvs_columns['categorical'] + wvs_columns['numerical'] + wvs_columns['ordinal']:
            if i in wvs_part_cat.columns.values.tolist():
                if i in wvs_drop_1:
                    wvs_part_cat = wvs_part_cat.drop(i, axis = 1)
                else:
                    labelencoder = wvs_labelencoders[i]
                    try:
                        wvs_part_cat[i] = labelencoder.transform(wvs_part_cat[i])
                    except:
                        new_values = list(set(wvs_part_cat[i].unique()) - set(labelencoder.classes_))
                        for value in new_values:
                            labelencoder.classes_ = np.append(labelencoder.classes_, value)
                        wvs_labelencoders[i] = labelencoder
                        wvs_part_cat[i] = labelencoder.transform(wvs_part_cat[i])
            elif i in wvs_part_num.columns.values.tolist():
                if i in wvs_drop_4:
                    wvs_part_num = wvs_part_num.drop(i, axis = 1)
                else:
                    wvs_part_num[i] = wvs_part_num[i].fillna(wvs_mean_4[i])
            else:
                if i in wvs_drop_5:
                    wvs_part_ord = wvs_part_ord.drop(i, axis = 1)
                else:
                    wvs_part_ord[i] = wvs_part_ord[i].fillna(wvs_mean_5[i]).astype('int64')
        wvs_part = pd.concat([wvs_part_cat, wvs_part_num, wvs_part_ord], axis=1)
        final_merge = pd.concat([hsbc_part, wvs_part], axis=1)

        save_params('wvs','wvs_labelencoders')

        return final_merge
    elif database == 'crrc':

        hsbc_part = merged_data.loc[:, :'Debt/Income']
        crrc_part = merged_data.loc[:, 'Debt/Income':].iloc[:, 1:]

        relevant_cat_cols = ['ACTCHORE', 'ACTRESDT', 'ACTCLEAN', 'ACTTHEA', 'ACTREST',
                             'ACTVLNT', 'ACTCOMM', 'ACTDNCH', 'ACTNRCHN', 'ACTPBLM', 'ACTSPET',
                             'ACTDON', 'HLTHRAT', 'CLRLABR', 'CLFRDAB', 'GENBREA', 'GENBRER', 'APTINHERT', 'POLDIRN',
                             'IMPISS1', 'IMPISS2',
                             'REVEXP1', 'REVEXP2', 'FREESPK', 'EMPLSIT', 'GETJOBF', 'NEWHHJOB', 'JOBLOST', 'RFAEDUC',
                             'RMOEDUC', 'FLMANDSC',
                             'EMIGRAT', 'MIGSHRT', 'OWNCOTV',
                             'OWNDIGC', 'OWNWASH', 'OWNFRDG',
                             'OWNAIRC', 'OWNCARS', 'OWNLNDP', 'OWNCELL',
                             'OWNPCELL', 'OWNCOMP', 'SAVINGS', 'DEBTSHH']
        relevant_ord_cols = ['TRUBANK', 'GALLTRU', 'DISCPRPR', 'FATEINLF', 'TRUEDUC', 'TRUCRTS', 'TRUARMY', 'TRUPARL',
                             'TRUEXEC',
                             'TRUPRES', 'TRUPOLI', 'TRUSTEU', 'QUALINF', 'FAIRTRT', 'JOBSARFN', 'FRQINTR', 'RELSERV',
                             'ECONSTN',
                             'NOMONFD', 'MONYTOT', 'SPENDMO', 'FOODDBT', 'UTILDBT', 'CURRUNG', 'FUTRUNG', 'RELCOND',
                             'MININCR']
        relevant_num_cols = ['HHSIZE', 'HHASIZE', 'IDEALNCH', 'NUMCOTV', 'NUMFRDG', 'NUMDIGC', 'NUMWASH', 'NUMAIRC',
                             'NUMCELL',
                             'NUMCARS', 'NUMPCELL', 'NUMCOMP']

        crrc_part_cat = crrc_part.loc[:, relevant_cat_cols]
        crrc_part_ord = crrc_part.loc[:, relevant_ord_cols]
        crrc_part_num = crrc_part.loc[:, relevant_num_cols]
        for i in relevant_cat_cols + relevant_ord_cols + relevant_num_cols:
            if i in relevant_cat_cols:
                if i in crrc_drop_1:
                    crrc_part_cat = crrc_part_cat.drop(i, axis = 1)
                else:
                    crrc_part_cat[i] = crrc_part_cat[i].fillna(crrc_max_1[i])
            elif i in relevant_num_cols:
                if i in crrc_drop_4:
                    crrc_part_num = crrc_part_num.drop(i, axis = 1)
                else:
                    crrc_part_num[i] = crrc_part_num[i].fillna(crrc_mean_4[i])
            else:
                if i in crrc_drop_5:
                    crrc_part_ord = crrc_part_ord.drop(i, axis = 1)
                else:
                    crrc_part_ord[i] = crrc_part_ord[i].fillna(crrc_mean_5[i]).astype('int64')
        crrc_part = pd.concat([crrc_part_cat, crrc_part_ord, crrc_part_num], axis=1)
        final_merge = pd.concat([hsbc_part, crrc_part], axis=1)

        return final_merge


def get_final_merge(new_data, new_data_preprocessed, database):
    if database == 'crrc':
        weights = np.array([8, 7, 8, 6, 7, 10, 10, 10, 8, 10, 2, 6])
    elif database == 'wvs':
        weights = np.array([2, 8, 8, 9, 6, 6, 5, 9, 7])
    mp = merge_prepare(new_data, new_data_preprocessed, database)
    hsbc_overlaps = mp.hsbc_overlaps()
    crrc_overlaps = mp.database_overlaps()
    hsbc_data = mp.hsbc_whole()
    crrc_data = mp.database_whole()

    directory = folder + 'Preprocessing/merge/' + database + '/'
    m = merge_overlaps(hsbc_data, crrc_data, hsbc_overlaps, crrc_overlaps, weights=weights, warm_start = True, directory = directory)
    merged_data = m.merge()

    final_merge = postprocess(merged_data, database)
    return final_merge


def get_feed_data(new_data, database):

    new_data = correct_transform(new_data)
    new_data_preprocessed = hsbc_preprocessing(new_data)
    relevant_columns = pd.read_csv(folder + 'Data/' + database + '/x_train.csv').columns.values.tolist()
    if database == 'crrc_wvs':
        final_merge_crrc = get_final_merge(new_data, new_data_preprocessed, 'crrc')
        final_merge_wvs = get_final_merge(new_data, new_data_preprocessed, 'wvs')

        final_merge = final_merge_crrc.copy()
        for col in final_merge_wvs.columns.values:
            final_merge[col] = final_merge_wvs[col]
    elif database == 'hsbc':
        final_merge = new_data_preprocessed.loc[:,:'Debt/Income']
    else:
        final_merge = get_final_merge(new_data, new_data_preprocessed, database)
    final_merge = final_merge.loc[:, relevant_columns]

    def fill_missings(data, categoricals):
        d = data.copy()
        d = d.replace([np.inf, -np.inf], np.nan)

        turnovers = ['Local Currency Average balance for last 1month (customer level)',
                     'Local Currency Average balance for last 6 months (customer level)',
                     'Credit turnover  for last 6 months (customer level)',
                     'Debit turnover for last 6 months (customer level)', 'Total turnover']
        for i in turnovers:
            if i in data.columns.values.tolist():
                d[i] = d[i].fillna(0)
        categoricals = list(set(categoricals) & set(d.columns.values.tolist()))
        with open(folder + 'Preprocessing/' + 'fill_cat_' + database + '.txt', 'rb') as file:
            fill_cat = pickle.load(file)
        for i in categoricals:
            d[i] = d[i].fillna(fill_cat[i])
        numericals = list(set(d.columns.values.tolist()) - set(categoricals))
        with open(folder + 'Preprocessing/' + 'fill_num_' + database + '.txt', 'rb') as file:
            fill_num = pickle.load(file)
        for i in numericals:
            d[i] = d[i].fillna(fill_num[i])
        return d

    with open(folder + 'Preprocessing/' + database + '_cat_cols.txt', 'rb') as file:
        behavioral_cat_features = pickle.load(file)
    with open(folder + 'Preprocessing/hsbc_cat_cols.txt', 'rb') as file:
        hsbc_cat_features = pickle.load(file)
    cat_features = np.unique(np.append(hsbc_cat_features, behavioral_cat_features)).tolist()

    feed_data = fill_missings(final_merge, categoricals=cat_features)

    return feed_data


class ThreshValidation:
    def __init__(self, model, x, y, neg_threshes, cv=5, n_jobs=-1):
        self.model = clone(model)
        self.x = x
        self.y = y
        self.cv = cv
        self.n_jobs = n_jobs
        self.neg_threshes = neg_threshes

    def set_seperation(self):
        if type(self.cv) == int:
            kf = KFold(n_splits=self.cv, random_state=42, shuffle=True)
        else:
            kf = self.cv
        splits = kf.split(self.x)

        self.splits = splits

    def train_model(self, test_index, train_index):
        model = clone(self.model)

        x_train = self.x.iloc[train_index]
        y_train = self.y[train_index]
        return model.fit(x_train, y_train)

    def evaluate(self, model, test_index, train_index):
        x_test = self.x.iloc[test_index]
        y_test = self.y[test_index]

        probas = model.predict_proba(x_test)[:, 1]

        if not isinstance(self.neg_threshes, Iterable):
            neg_thresh = self.neg_threshes
            best_thresh = best_threshold(y_test, probas, neg_thresh=neg_thresh)
            return best_thresh
        else:
            best_threshes = []
            for neg_thresh in self.neg_threshes:
                best_threshes += [best_threshold(y_test, probas, neg_thresh=neg_thresh)]
            return best_threshes

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

        if isinstance(self.threshes[0], Iterable):
            best_threshes = pd.DataFrame(np.array(self.threshes)).mean(axis=0)
            threshes = {}
            for i, neg_thresh in enumerate(self.neg_threshes):
                threshes[neg_thresh] = best_threshes.iloc[i]
            self.threshes = threshes
            return self.threshes
        else:
            return self.threshes


def get_predictions(model, data, neg_thresh, threshes):
  probas = model.predict_proba(data)[:,1]
  predictions = np.where(probas < threshes[neg_thresh], 0, 1)
  return predictions