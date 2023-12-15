### Imports
## General
import os
import glob
import math
import pdb
import wheel
import numpy as np
## Pandas
import pandas as pd
from pandas.tseries.offsets import DateOffset
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.plotting import lag_plot
## Date-management imports
from datetime import date
from datetime import datetime
# pip install python-dateutil
from dateutil import relativedelta
## Plots
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
## Statistics and forecasts
# Scipy
import scipy.stats
# Scikit-learn
import sklearn
from sklearn.linear_model import LinearRegression
# Statsmodels
import statsmodels.api as sm
# Auto-arima
import pmdarima as pm
from pmdarima.arima.utils import ndiffs
from pmdarima.model_selection import train_test_split
from pmdarima.arima.stationarity import ADFTest, PPTest, KPSSTest
from pmdarima.arima.utils import nsdiffs
## Popular time series packages
# import sktime
# it might be useful to upgrade plotly before importing prophet
# import prophet
# to install: pip install git + https: // github.com / RJT1990 / pyflux
# import pyflux
# import flint
# import kats (requires BLAS and LAPACK, better to use with miniconda)
# import tsfresh (requires numba to have been made compatible with python 3.11)
# import darts (requires numba to have been made compatible with python 3.11)
# import pastas (requires numba to have been made compatible with python 3.11)
# import arrow
# import orbit
## H2O
# H2O requires 'requests', 'tabulate' and 'future' to be installed
# Use pip install -f https://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o to install
# import h2o
## Functions
from old_functions_to_be_migrated.utility_functions import *
## Non-timeseries package here for reference
# import ChannelAttribution
# import ortools
# import dalex

## To get package site in pycharm:
# import site
# print(site.getsitepackages())

## Reference links to get all manner of datasets:
# https://data.gov/
# https://www.bls.gov/data/
# https://data.europa.eu/en
# https://www.data.gouv.fr/fr/
# https://www.data.gov.uk/
# https://odre.opendatasoft.com
# https://www.kaggle.com/datasets
#

## Data preparation and loading

folder_path = r'D:\VIANNEY\DEV\TIME_SERIES\SRC\TIMES_SERIES\data'
folders = {'national': r'\eco2mix-national-tr.csv', 'regional': r'\eco2mix-regional-tr.csv',
            'metropolitan': r'\eco2mix-metropoles-tr.csv'}
folder_files = glob.glob(os.path.join(folder_path, '*csv'))
print(f'Files: {folder_files}')
# Checking 'Date', 'Heure' and 'Date-Heure' columns shows that the time index has not been rebuilt
# And that the time is in CET
# TODO: Handle missing values a bit better to not lose so many observations
# TODO: Debug add_data_columns? (might not be worth the effort)
datasets = {'national': []} #, 'regional': [], 'metropolitan': []}
for key in datasets.keys():
    # Loading raw data (taken from eco2mix)
    if folder_path + folders[key].replace('.csv', '_mod.csv') not in folder_files:
        datasets[key] = pd.read_csv(folder_path + folders[key], sep = ';')
        df = datasets[key].copy()
        # Creating time index
        df = df.rename(columns = {'Date - Heure': 'date', 'Date': 'Jour'})
        df = df.set_index('date', drop = True).sort_index()
        df.index = pd.to_datetime(df.index, utc = True)
        df.index = df.index.tz_convert('CET')
        # Handling columns with special values
        if 'Périmètre' in df.columns and len(df['Périmètre'].unique()) == 1 and df['Périmètre'][0] == 'France':
            df = df.drop('Périmètre', axis = 1)
        if len(df['nature'].unique()) == 1 and df['nature'][0] == 'Données temps réel':
            df['nature'] = 'R'
        if 'Pompage_(MW)' in df.columns:
            df['Pompage_(MW)'] = [p.replace('-', '0') for p in df['Pompage_(MW)']].astype(float)
        df.columns = [col.replace(' - ', '_').replace('. ', '_').replace(' ', '_') for col in df.columns]
        # Excluding empty, redundant or otherwise non-informative columns
        throw_cols = [col for col in df.columns if ('Ech_comm' in col) | ('offshore' in col) |
                       ('terrestre' in col) | ('batterie' in col) | ('68' in col)]
        df = df.drop(throw_cols, axis = 1)
        df = df.dropna()
        # Adding utility date columns
        # df = add_date_columns(df, 'date', value_column = 'Consommation_(MW)', change_index = True, tz = 'CET')
        for period in ['year', 'month', 'day', 'hour']:
            df[period] = eval(f'df.index.{period}')
        df['year_month'] = df.index.strftime('%Y-%m')
        df['exact_day'] = df.index.strftime('%Y-%m-%d')
        print(f'Current key: {key}')
        # df['season'] = [current_season(ind) for ind in df.index] #bugged? maybe because of datetime name ambiguity
        df['weekday'] = [ind.weekday() for ind in df.index]
        df['is_weekend'] = [ind.weekday() > 4 for ind in df.index]
        # Saving prepared data
        df.to_csv(folder_path + folders[key].replace('.csv', '_mod.csv'), sep = ';', encoding = 'latin-1')
        print(f'Saving {key} modified dataset to {folder_path}')
        datasets[key] = df
        del df
    else:
        # Loading preapred data and recreating time index
        print(f'Loading {key} modified dataset from {folder_path}')
        datasets[key] = pd.read_csv(folder_path + folders[key].replace('.csv', '_mod.csv'), 
                                    sep = ';', encoding = 'latin-1')
        datasets[key] = datasets[key].set_index('date', drop = True)
        datasets[key].index = pd.to_datetime(datasets[key].index, utc = True)
        datasets[key].index = datasets[key].index.tz_convert('CET')
        # Printing columns dtypes and number of missing values to check if preparations have worked
        print(f'Columns types in {key} data:\n {datasets[key].dtypes} \n Missing values in {key} data columns: \n {datasets[key].isna().sum()}')

## Graphical exploration

# if not os.path.isdir(f'{folder_path}\\Figures'):
#     os.mkdir(f'{folder_path}\\Figures')

for key in datasets.keys():
    df = datasets[key]
    # Plotting and saving the variable of interest
    plt.plot(df['consommation'])
    plt.savefig(f'{folder_path}\\Figures\\{key}_consommation.png')
    plt.close()
    # Plotting the variable of interest on different timescales and saving it
    years = df.index.year.unique()
    for year in years:
        # Year-specific plot
        df_year = df.loc[df.index.year == year]
        plt.plot(df_year['consommation'])
        plt.savefig(f'{folder_path}\\Figures\\{key}_{year}_consommation.png')
        # plt.close()
        timescale_plots(df_year, 'consommation', axes = [4, 6],
                    timescales = {'unit': 'day', 'graph': 'month', 'global': 'year'}, 
                    save_folder = f'{folder_path}\\Figures\\{key}_{year}_consommation_yearly.png')
    # Yearly plot on monthly scale
    timescale_plots(df, 'consommation', axes = [4, 6],
                    timescales = {'unit': 'month', 'graph': 'year', 'global': 'all'}, 
                    save_folder = f'{folder_path}\\Figures\\{key}_consommation_yearly.png')
    
