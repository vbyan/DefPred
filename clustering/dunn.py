import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from CFS import helpers
import clustering
os.chdir('/Users/Artavazd Maranjyan/Desktop/Clustering')
Data = pd.read_csv('WVS.csv')
Data2=helpers.row_selection(Data,percent = 10)
cols = ['H_URBRURAL'  ,'B_COUNTRY'  ,'Q264'  ,'Q144' ,'Q1'  ,'Q19'  ,'Q263'  ,'Q260'  ,'doi'  ,'AUTONOMY'  ,'RESEMAVALWGT'  ,'Q17'  ,'Q265' ,'Refugeesorigin'  ,'co2emis'  ,'landWB' ,'I_PSU']
data = Data2.loc[:,cols]
clustering.GMM_clustering(data, 30,'SCFP',include = 'all')

