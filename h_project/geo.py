import numpy as np
import warnings
warnings.filterwarnings('ignore')
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="User", timeout = None)


def get_correct_address(bad_address):
    if isinstance(bad_address,str):
        pass
    else:
        bad_address = ''
    split_list = bad_address.split(' ')
    for i in range(len(split_list)):
        split_list[i] = split_list[i].replace(',','').replace('/','').replace('.','')
    split_list = ['STREET' if i in ['STR','ALLEY'] else i for i in split_list]
    split_list = ['AVE' if i == 'AVENUE' else i for i in split_list]
    split_list = ['VILLAGE' if i == 'VLG' else i for i in split_list]
    split_list = ['' if i in ['TH','DISTR','DIST','DSTR'] else i for i in split_list]
    split_list = ['' if len(i) == 1 else i for i in split_list]
    for word in ['APT',',BLD','BLD','HOUSE','FLOOR','RA','-ND','LANE','RD','BLOCK','MICROREGION','REGION','REG','ND','MICRODIST','CITY','MIKROREGION','APPT','NR','MIKRODISTRICT','SHRJ','COMMUNITY']:
        try:
            split_list.remove(word)
        except:
            pass
    correct_text = ''
    for i in split_list:
        correct_text += ' ' + i
    return correct_text

def get_geolocation(x):
    location = geolocator.geocode(x, timeout = None)
    return location
def all_variants(x):
    x = get_correct_address(x)
    variants = [] + [x]
    X = x.replace('STREET','')
    variants += [X]
    split_list = x.split(' ')
    X = ''
    for i in split_list[:-1]:
        X += ' ' + i
    variants += [X]
    while True:
        try:
            split_list.remove('')
        except:
            break
    if (len(split_list) == 3)&('STREET' not in split_list)&('AVENUE' not in split_list)&('VILLAGE' not in split_list):
        first_item = split_list[0]
        second_item = split_list[1]
        third_item = split_list[2]
        X = second_item + ' ' + first_item + ' ' + third_item
        variants += [X]
    try:
        street_name = split_list[split_list.index('STREET') - 1]
        X = x.replace(street_name,'').replace('STREET','')
        variants += [X]
    except:
        try:
            street_name = split_list[split_list.index('STREET') - 1]
            if street_name.__contains__('C'):
                X = x.replace('C','TS')
                variants += [X]
            elif street_name.__contains__('G'):
                X = x.replace('G','H')
                variants += [X]
        except:
            pass
    return variants
def try_until_success(x):
    variants = all_variants(x)
    for variant in variants:
        location = get_geolocation(variant)
        if location != None:
            return location[1]
            break
        else:
            continue


def get_street_name(address):
    if is_yerevan(address):
        address = Get_correct_address(address)
        split_list = address.split(' ')
        split_list = ['ST' if i in ['STR', 'ALLEY', 'ROAD', 'ATR', 'STT', 'STREET'] else i for i in split_list]
        split_list = ['AVE' if i == 'AVENUE' else i for i in split_list]
        split_list = ['HIGHWAY' if i in ['HWY', 'HGW'] else i for i in split_list]
        split_list = ['RA' if i in ['R A', 'ARMENIA'] else i for i in split_list]
        if 'ST' in split_list:
            street_index = split_list.index('ST')
            street_name = split_list[street_index - 1]
            entire_address = street_name + ' STREET YEREVAN'
            try:
                location = Get_geolocation(entire_address)[1]
                return location
            except:
                street_name = modify(street_name)
                entire_address = street_name + ' STREET YEREVAN'

                try:
                    location = Get_geolocation(entire_address)[1]
                    return location
                except:
                    return np.nan

        elif 'AVE' in split_list:
            street_index = split_list.index('AVE')
            street_name = split_list[street_index - 1]
            entire_address = street_name + ' AVE YEREVAN'
            try:
                location = Get_geolocation(entire_address)[1]
                return location
            except:
                street_name = modify(street_name)
                entire_address = street_name + ' AVE YEREVAN'
                try:
                    location = Get_geolocation(entire_address)[1]
                    return location
                except:
                    return np.nan

        elif 'HIGHWAY' in split_list:
            street_index = split_list.index('HIGHWAY')
            street_name = split_list[street_index - 1]
            entire_address = street_name + ' HIGHWAY YEREVAN'
            try:
                location = Get_geolocation(entire_address)[1]
                return location
            except:
                street_name = modify(street_name)
                entire_address = street_name + ' HIGHWAY YEREVAN'
                try:
                    location = Get_geolocation(entire_address)[1]
                    return location
                except:
                    return np.nan

        elif 'RA' in split_list:
            ra_index = split_list.index('RA')
            new_list = split_list[:ra_index + 1]
            entire_address = ''
            for i in new_list:
                entire_address += ' ' + i
            try:
                location = Get_geolocation(entire_address[1:])[1]
                return location
            except:
                return np.nan

        else:
            return np.nan
    else:
        return np.nan


def Get_correct_address(bad_address):
    bad_address = bad_address.replace(',', ' ').replace('/', ' ').replace('.', ' ')
    split_list = bad_address.split(' ')
    split_list = ['ST' if i in ['STR', 'ALLEY'] else i for i in split_list]
    split_list = ['AVE' if i == 'AVENUE' else i for i in split_list]
    split_list = ['VILLAGE' if i == 'VLG' else i for i in split_list]
    split_list = ['' if i in ['TH', 'DISTR', 'DIST', 'DSTR'] else i for i in split_list]
    split_list = ['' if len(i) == 1 else i for i in split_list]
    for word in ['APT', ',BLD', 'BLD', 'HOUSE', 'FLOOR', '-ND', 'LANE', 'RD', 'BLOCK', 'MICROREGION', 'REGION', 'REG',
                 'ND', 'MICRODIST', 'CITY',
                 'MIKROREGION', 'APPT', 'NR', 'MIKRODISTRICT', 'SHRJ', 'COMMUNITY', 'BUILDING', 'AREAS', 'MICRO',
                 'OFFICE', 'GOVERNMENT', 'GOVERNMENTS',
                 'ROOM', 'AREA', 'POGH', 'FLR', 'BUILD','UNDP']:
        try:
            split_list.remove(word)
        except:
            pass
    correct_text = ''
    for i in split_list:
        correct_text += ' ' + i
    return correct_text[1:]





def is_yerevan(address):
    try:
        if (address.__contains__('YEREVAN RA')) | (address.__contains__('YEREVAN R A')) | (
        address.__contains__('YEREVAN, RA')) | (address.__contains__('YEREVAN ARMENIA')):
            return True
        else:
            return False
    except:
        pass


def Get_geolocation(x):
    location = geolocator.geocode(x, timeout=None)
    return location


def modify(street_name):
    if street_name[-3:] == 'IAN':
        street_name = street_name[:-3] + 'YAN'
    elif street_name[-4:] == 'YANC':
        street_name = street_name[:-4] + 'YAC'
    elif street_name[-4:] == 'YA':
        street_name = street_name[:-4] + 'YAN'
    elif street_name[-3:] == 'YAC':
        street_name = street_name[:-3] + 'YATS'
    elif street_name[-3:] == 'YUN':
        street_name = street_name[:-3] + 'YAN'
    elif street_name[-3:] == 'YAN':
        street_name = street_name[:-3] + 'YUN'
    else:
        street_name = street_name + 'I'
    return street_name