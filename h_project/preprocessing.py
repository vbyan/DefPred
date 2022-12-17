from dateutil import parser
import pandas as pd

from geo import *
from tqdm.notebook import tqdm
from colorama import Fore
import string
import xlwings as xw
import pickle
import os

if os.path.exists('/content/gdrive/MyDrive/Loan_Scoring_Third_Phase/HSBC/Preprocessing/'):
    folder = '/content/gdrive/MyDrive/Loan_Scoring_Third_Phase/HSBC/Preprocessing/'
else:
    folder = 'G:\My Drive\Loan_Scoring_Third_Phase\HSBC/Preprocessing/'




def check_date(date):
    if date.isdigit():
        return date
    else:
        try:
            parsed_date = parser.parse(date)
            day = str(parsed_date.day)
            month = str(parsed_date.month)
            year = str(parsed_date.year)

            if len(day) == 1:
                day = '0' + day
            if len(month) == 1:
                month = '0' + month

            correct_date = year + month + day
            correct_date = int(correct_date)

            return correct_date
        except:
            return np.nan


check_date('Oct 04 2021')
# entire pipeline to fully preprocess the HSBC raw data
def read_locked_xlsx(path_to_file, password = None):
    def get_entire_sheet(sheet,start = 'A2'):
        alphabet = string.ascii_uppercase
        last_row = sheet.range(start).end('down').row
        last_col_index = sheet.range(start).end('right').column - 1
        col_name = alphabet[last_col_index]
        last_cell = col_name + str(last_row)
        return sheet[start + ':' + last_cell].options(pd.DataFrame, index=False, header=True).value

    wb = xw.Book(path_to_file, password = password)
    sheet1 = wb.sheets['Customer data']
    sheet2 = wb.sheets['Facility data']
    sheet3 = wb.sheets['Matched with Initial Dataset']

    customer_data = get_entire_sheet(sheet1)
    facility_data = get_entire_sheet(sheet2)
    id_data = get_entire_sheet(sheet3, start = 'A1')
    app = xw.apps.active


    entire_data = pd.merge(facility_data, customer_data, left_on='UID',
                           right_on='UID', how='right')

    new_id_transformer = {}
    for i in id_data.index.values:
        seq_value = id_data['Seq UID'].loc[i]
        relevant_value = id_data['Previous UID'].loc[i]
        new_id_transformer[seq_value] = relevant_value

    entire_data['UID'] = entire_data['UID'].replace(new_id_transformer)
    app.quit()

    return entire_data


def hsbc_preprocessing(data):

    # There are some features where nan values are more intuitively correct than the initial ones
    # Actually this part is useless, because in order to combine several models (RF,XGB, CAT) nan values were replaced
    def replace_with_nan(series, data):
        summ = data.loc[:,
               'Local Currency Average balance for last 1month (customer level)':'Debit turnover for last 6 months (customer level)'].sum(
            axis=1)
        zeros_indices = summ[summ == 0].index.values.tolist()
        series[zeros_indices] = np.nan
        return series

    # Transforms the date features in string format into date format
    def parse(x):
        x = str(x)
        date_format = x[:4] + '-' + x[4:6] + '-' + x[6:8]
        try:
            return parser.parse(date_format)
        except:
            return np.nan

    # From here on the function carries out the main preprocessing steps
    # First it reads the sheets of the initial Excel file defined by the path to it
    with open(folder + 'label_encoders.txt', 'rb') as file:
        label_encoders = pickle.load(file)

    entire_data = data.copy()

    # Here is the preprocessing of features related to customer accounts
    entire_data.replace({'': np.nan, None: np.nan, 'None': np.nan}, inplace=True)
    balance_cols = entire_data.loc[:,
                   'Local Currency Average balance for last 1month (customer level)':'Debit turnover for last 6 months (customer level)'].columns.values.tolist()
    entire_data.loc[:, balance_cols] = entire_data.loc[:, balance_cols].apply(
        lambda x: replace_with_nan(x, entire_data))
    #entire_data['Reason of decline'] = entire_data['Reason of decline'].fillna('No reason')
   #entire_data['Current limit  (only for reviews)'] = entire_data['Current limit  (only for reviews)'].fillna(
        #entire_data['Approved\nlimit'])

    # Preprocessing of salary-related features
    entire_data['Salary periodicity'] = entire_data['Salary periodicity'].fillna('M')
    entire_data = entire_data.drop('Previous Employer', axis=1)
    entire_data['Other income periodicity'][entire_data['Other income (in AMD)'] == 0] = 'M'
    entire_data['Other income weights'] = entire_data['Other income periodicity'].replace(
        {'M': 12, 'H': 3, 'Q': 3, 'X': 3, 'Y': 1})
    entire_data['Other income weighted'] = entire_data['Other income (in AMD)'] * entire_data['Other income weights']
    entire_data['Salary weights'] = entire_data['Salary periodicity'].replace(
        {'M': 12, 'Y': 1, 'X': 12, 'H': 24, 'A': 24, 'L': 12, 'Q': 3})
    print(Fore.GREEN + '1. Unstructured data is handled')
    print(Fore.BLACK + '----------------------------------------------------------------------------------------')

    entire_data['Salary weighted'] = entire_data['Salary (in AMD)'] * entire_data['Salary weights']
    entire_data = entire_data.drop(['Salary (in AMD)', 'Other income (in AMD)', 'Salary periodicity',
                                    'Other income periodicity', 'Salary weights', 'Other income weights',
                                    'Car Ownership Status Yes/No'], axis=1)

    # Bringing some string features to the same format
    entire_data['Loan/Card/Overdraft Type'] = entire_data['Loan/Card/Overdraft Type'].replace(
        {'PAL Refinancing': 'PAL REFINANCING',
         'PAL refinancing': 'PAL REFINANCING', 'Credit card Limit review': 'Credit card Limit Review'})

    with open(folder + 'hsbc_cat_cols.txt', 'rb') as file:
        cat_features = pickle.load(file)

    entire_data.loc[:, cat_features] = entire_data.loc[:, cat_features].astype(str).applymap(
        lambda x: x.upper()).replace('NAN', np.nan)

    # LabelEncoding using the labelencoders trained on entire data
    # If new values are detected in upcoming datasets code should be reviewed and new values should be added
    entire_data['Work address original'] = entire_data['Work Address']
    entire_data['Residential address original'] = entire_data['Residential Address']

    for col in cat_features:
        try:
            entire_data[col] = label_encoders[col].transform(entire_data[col])
        except:
            try:
                new_values = list(set(entire_data[col].unique()) - set(label_encoders[col].classes_))
                for value in new_values:
                    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, value)
            except:
                pass

    with open(folder + 'label_encoders.txt', 'wb') as file:
        pickle.dump(label_encoders,file)

    # Preprocessing of date-related columns
    date_columns = ['Applic\ndate', 'Relationship start date (with the Bank)',
                    'Employed since (current employer)']
    for column in date_columns:
        d = entire_data[column].dropna().astype(str)
        String = d.apply(lambda x:check_date(x)).astype(str)
        String[String.str.len() < 8] = '20190101'
        String = String.astype('int64')
        entire_data[column] = String
    #entire_data['Approval date'].loc[entire_data['Approval date'][
        #(entire_data['Approval date'] - entire_data['Applic\ndate']) < 0].index.values.tolist()] = entire_data[
        #'Applic\ndate']

    entire_data['Applic\ndate\date'] = entire_data['Applic\ndate'].apply(lambda x: parse(x))
    #entire_data['Approval date\date'] = entire_data['Approval date'].apply(lambda x: parse(x))

    #entire_data['Pending days'] = (entire_data['Approval date\date'] - entire_data['Applic\ndate\date']).astype(
        #str).str[:-5].replace('', np.nan).astype('float64')
   # entire_data['Pending days'][entire_data['Pending days'] < 0] = 0

    entire_data['Employed since (current employer)\date'] = entire_data['Employed since (current employer)'].apply(
        lambda x: parse(x))

    entire_data['Experience(Current employer)'] = (entire_data['Applic\ndate\date'] - entire_data[
        'Employed since (current employer)\date']).apply(lambda x: x.total_seconds()) / 31536000
    entire_data['Total Experience'] = entire_data['Experience(Current employer)'] + entire_data[
        'Duration with previous employer (years)']

    entire_data = entire_data.drop(
        ['Applic\ndate\date', 'Employed since (current employer)\date'], axis=1)
    print(Fore.GREEN + '2. Date-related columns are handled')
    print(Fore.BLACK + '----------------------------------------------------------------------------------------')

    # Engineering new features based on the ones from dataset
    entire_data['Total income'] = entire_data['Salary weighted'] * 12 + entire_data['Other income weighted']
    entire_data['Total turnover'] = entire_data['Credit turnover  for last 6 months (customer level)'] + entire_data[
        'Debit turnover for last 6 months (customer level)']
    entire_data['Debt/Income'] = entire_data['Amount/AMD'] / entire_data['Total income']
    print(Fore.GREEN + '3. New features are engineered')
    print(Fore.BLACK + '----------------------------------------------------------------------------------------')
    print(Fore.GREEN + '4. Only coordinates are left')
    print(Fore.YELLOW + 'They are being tansformed right now')
    print(Fore.YELLOW + 'This may take a while')
    tqdm.pandas()

    # Getting work address and residential address coordinates
    # This is the main time-consuming part
    residential_address_coordinates = entire_data['Residential address original'].progress_apply(
        lambda x: try_until_success(x))
    print(Fore.BLACK + '----------------------------------------------------------------------------------------')
    work_address_coordinates = entire_data['Work address original'].progress_apply(lambda x: get_street_name(x))

    print(Fore.GREEN + 'All done')
    print(Fore.BLACK + '----------------------------------------------------------------------------------------')
    entire_data['Residential long'] = residential_address_coordinates.str[1]
    entire_data['Residential lat'] = residential_address_coordinates.str[0]
    entire_data['Work long'] = work_address_coordinates.str[1]
    entire_data['Work lat'] = work_address_coordinates.str[0]
    entire_data = entire_data.drop(['Residential address original', 'Work address original'], axis=1)
    entire_data = entire_data.replace([np.inf, -np.inf], np.nan)

    entire_data['UID'] = entire_data['UID'].fillna(0).astype('int64')

    # Rearranging the columns to match the required arrangement
    with open(folder + 'columns_sequence.txt', 'rb') as file:
        columns_sequence = pickle.load(file)
    entire_data = entire_data.loc[:, columns_sequence]

    return entire_data
