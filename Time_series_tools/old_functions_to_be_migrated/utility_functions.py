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
# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
# Seaborn
import seaborn as sns
# Plotly
# import plotly.express as px
## Data reading
# import openpyxl
# from parse import *
## Statistics and forecasts
# Scipy
import scipy.stats as sci
import scipy.signal as sig
# Scikit-learn
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
# Statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.x13 import x13_arima_analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
# Auto-ari
import pmdarima as pm
from pmdarima.arima.utils import ndiffs
from pmdarima.model_selection import train_test_split
from pmdarima.arima.stationarity import ADFTest, PPTest, KPSSTest
from pmdarima.arima.utils import nsdiffs
## Popular time series packages
# import sktime
# # it might be useful to upgrade plotly before importing prophet
# import prophet
# # to install: pip install git+https://github.com/RJT1990/pyflux
# import pyflux
# import flint
# # import kats (requires BLAS and LAPACK, better to use with miniconda)
# # import tsfresh (requires numba to have been made compatible with python 3.11)
# # import darts (requires numba to have been made compatible with python 3.11)
# # import pastas (requires numba to have been made compatible with python 3.11)
# import tsfel
# import arrow
# import orbit
# ## H2O
# # H2O requires 'requests', 'tabulate' and 'future' to be installed
# # Use pip install -f https://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o to install
# import h2o
# ## Non-timeseries package here for reference
# import ChannelAttribution
# import ortools
# import dalex

# To get package site in pycharm:
import site
print(site.getsitepackages())

## Utility functions

# General utilities
# Random color generation function
def random_colors(n):
    colors = []
    for color in range(n):
        r = np.round(np.random.rand(), 1)
        g = np.round(np.random.rand(), 1)
        b = np.round(np.random.rand(), 1)
    colors.append([r, g, b])
    return colors

# Iterable nature verification function
def is_iter(obj):
    iterable = True
    try:
        iter(obj)
    except TypeError:
        iterable = False
    return iterable

## Time index utilities
# Season determination function
def current_season(now, y = 2000): # y is a dummy leap year
    seasons = [('winter', (date(y, 1, 1), date(y, 3, 20))),
               ('spring', (date(y, 3, 21), date(y, 6, 20))),
               ('summer', (date(y, 6, 21), (y, 9, 22))),
               ('autumn', (date(y, 9, 23), (y, 12, 20))),
               ('winter', (date(y, 12, 21), (y, 12, 31)))]
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year = y)
    return next(season for season, (start, end) in seasons if start <= now <= end)

# Time offset function
def offset_data(data, offsets, diff = True):
    df_offset = data.copy()
    times = {'y': 'years', 'm': 'months', 'd': 'days', 'h': 'hours', 's': 'seconds'}
    for key in offsets.keys():
        step = key
        if len(key) == 1:
            step = times[key]
        if step[-1] != 's':
            step += 's'
        df_offset.index += eval(f'DateOffset({step} = {offsets[key]})')
    if diff:
        return data[1:] - df_offset[:-1]
    else:
        return df_offset

# Redundant date handling function
def time_redundancies(data, threshold = 1, col_select = None):
    if col_select is not None:
        df = pd.DataFrame(pd.to_datetime([]))
        double_dates = df.index
        for sel in data[col_select].unique():
            data_sel = data.query(f'{col_select} == @sel')
            dates, dates_counts = np.unique(data_sel.index, return_counts = True)
            double_dates.append(dates[dates_counts > threshold])
    else:
        dates, dates_counts = np.unique(data.index, return_counts = True)
        double_dates = dates[dates_counts > threshold]
    return data.loc[data.index.isin(double_dates)].copy()

# Timezone management function
""" Currently only UTC and CET timezones are fully supported
The "value_corrections" argument is a dict giving for each column the corrective multiplicative factor to apply to ambiguous hours """
def manage_timezone(data, value_corrections = {},  tz = 'UTC', nonexistent = 'shift_forward'):
    info = (data.index.tzinfo is not None)
    ambig_times = {'CET': [(ind.month == 10) & (ind.day > 24) & (ind.weekday() == 6) and (ind.hour == 2) for ind in data.index], 
                   'UTC': 'raise'}
    if tz not in ['UTC', 'CET']:
        ambig_times[tz] = 'raise'
    if info and str(data.index.tzinfo) != tz:
        data.index = data.index.tz_localize(None).tz_localize(tz, nonexistent = nonexistent, ambiguous = ambig_times[tz])
        if len(value_corrections.keys()) > 0:
            for key in value_corrections.keys():
                data.loc[ambig_times[tz], key] = data.loc[ambig_times[tz], key]*value_corrections[key]
    elif not info: 
        data.index = data.index.tz_localize(None).tz_localize(tz, nonexistent = nonexistent)
    return data

# Timestep management/expanding function
def expand_index(data, old_period = 'infer', new_period = 'H', fill_method = 'ffill'):
    if old_period == 'infer':
        old_period = pd.infer_freq(data.index) 
    data = data.to_period(old_period).resample(new_period).fillna(method = fill_method)
    return data

# Merging and grouping function for timeseries
def aggregate_date(data, cols_agg, agg_functions, index_format = None, date_column = 'date', period = None, tz = None):
    if isinstance(cols_agg, str):
        cols_agg = [cols_agg]
    if index_format is not None:
        if index_format == 'default':
            cols_agg = [data.index] + cols_agg
        else:
            cols_agg = [data.index.strftime(index_format)] + cols_agg
    data = data.groupby(cols_agg).agg(agg_functions)
    data = data.reset_index()
    data = data.set_index(date_column, drop = True)
    utc = (tz is not None)
    data.index = pd.to_datetime(data.index, utc = utc)
    if period is not None:
        data.index = data.index.to_period(period)
    return data

# Adding utility columns and handling time inddex
# !!! DO NOT USE Currently causes unsolved problems
def add_date_columns(data, date_column = 'date', value_column = 'value', change_index = False, tz = 'UTC', value_corrections = {}):
    df = data.copy()
    utc = (tz is not None)
    if isinstance(df.index[0], datetime) and not change_index:
        df.index = pd.to_datetime(df.index, utc = utc)
        if date_column in df.columns:
            df = df.drop(date_column, axis = 1)
    else:
        if date_column in df.columns: 
            dates = df[date_column]
            df = df.drop(date_column, axis = 1)
        elif date_column == 'index':
            dates = df.index
        else: 
            print(f'Cannot locate {date_column} in the data columns or row indices, returning data unchanged')
            return df
        if not isinstance(dates[0], datetime):
            df = df.reset_index(drop = True)
            dates = pd.to_datetime(dates, utc = utc)
            # pdb.set_trace()
        df[date_column] = dates
        df = df.set_index(date_column, drop = True).sort_index()
    df = manage_timezone(df, tz = tz, value_corrections = value_corrections)
    for period in ['year', 'month', 'day', 'hour']:
        df[period] = eval(f'df.index.{period}')
    df['year_month'] = df.index.strftime('%Y-%m')
    df['exact_day'] = df.index.strftime('%Y-%m-%d')
    df['season'] = [current_season(ind) for ind in df.index]
    df['weekday'] = [ind.weekday() for ind in df.index]
    df['is_weekend'] = [ind.weekday() > 4 for ind in df.index]
    return df

# General time index management function
# def manage_index(time_series, to_index = True, drop = True, timezone = 'UTC', freq = 'H', timestep = 'H', 
#                  time_type = 'timestamp', expand = False, aggregate = False, complete = False):

## Plot functions

# Basic function for multiplot .png file
""" The "series_to_plot" argument is a dict with plot labels as keys and dataframes as values
The "graph_id" argument is a dict with a single key and a single series as value
For each element of the "graph_id" dict value a graph is plotted
For each graph, one or several curves are plotted
The curves are taken from the "series_to_plot" dict values
The label for each curve is the corresponding "series_to_plot" dict key
The result is a .png file containing n x m graphs where axes = (n, m)
Outputs a single .png file containing the multicurve graphs corresponding to all values of "graph_id"
This file is then saved in "save_folder" (unless it is None) """
def multiplot(series_to_plot, graph_id, x_col, y_col, suptitle, axes,
              figsize = (18, 22), colors = 'random', titles = 'default', loc = 'upper right', save_folder = None):
    # Creating general plot grid and setting plot parameters
    n = axes[0]
    m = axes[1]
    fig, axs = plt.subplots(n, m, figsize = figsize, squeeze = False)
    fig.suptitle(suptitle)
    if colors == 'random':
        colors = random_colors(len(series_to_plot.keys()))
    # Setting graph-specific parameters
    cat_name = list(graph_id.keys())[0]
    cat_values = list(graph_id.values())[0]
    if titles == 'default':
        titles = [f'{cat}' for cat in cat_values]
    # Plotting a graph for each value in the 'cat_values' series
    for i, cat in enumerate(cat_values):
        print(f'Plotting graph for {cat_name}: {cat}')
        # Plotting a curve for each dataframe in 'series_to_plot'
        for j, key in enumerate(series_to_plot.keys()):                
            df = series_to_plot[key].query(f'{cat_name} == @cat')
            if x_col == None:
                df = df.sort_index()
                axs[i//m, i%m].plot(df.index, df[y_col].mean(), label = key, color = colors[j])
            else:
                axs[i//m, i%m].plot(df[x_col].sort_values().unique(), df.groupby([x_col])[y_col].mean(), label = key, color = colors[j])
        # Giving a title and legend to the graph
        axs[i//m, i%m].set_title(titles[i])
        axs[i//m, i%m].legend(loc = loc)
    # Saving the multigraph in a .png file if required
    if save_folder is not None:
        os.makedirs(os.path.dirname(save_folder), exist_ok = True)
        plt.savefig(save_folder)
    plt.close()

# Single timescale plot function
""" Uses the multiplot function above to draw one or several timeseries at one timescale compared to another
To be used on data prepared with the 'add_date_columns' functions
The typical timescales are day, week, month, quarter, and year
The series can be be separated in periods with a plot each, or be plotted as the period varies on a single graph
The 'files_column' argument is the name of the column containing the condition corresponding to each .png file
The 'value_column' argument is the quantity to plot
The 'series_column' the name of the column containing the identifier of the series for which to plot 'value_column' 
(if only one series is to be plotted, set 'series_column' equal to the plot label of the variable to be plotted)
The 'timescales' argument is a dict whose value are the names of, respectively:
- 'unit' : the period between two consecutives observations of a graph (if None, this will be the natural period in the data)
- 'graph': the total length of time represented in each graph
- 'global' : the total period represented by all the graphs in the file
The 'period_column' argument is optional and contains  """
def timescale_plots(data, value_column, files_column = None, series_column = None, axes = 'default', timescales = 'default', period_column = None, 
                   figsize = (18, 22), colors = 'random', loc = 'upper right', save_folder = None):
    df = data.copy()
    # Setting general time parameters
    if timescales == 'default':
        timescales = {'unit': 'hour', 'graph': 'day', 'global': 'month'}
    t_graph = timescales['graph']
    t_unit = timescales['unit']
    t_global = timescales['global']
    # If there is more than one global period in the data, we select only the data corresponding to the first global period
    if t_global != 'all':
        global_periods = eval(f'df.index.{t_global}.unique()')
        n_global = len(global_periods)
        if  n_global > 1:
            df = df.loc[eval(f'df.index.{t_global} == {global_periods[0]}')]
        t_global = 'a ' + t_global + ' period'
    else:
        t_global = 'all observations'
    # For each unique value in files_column, we plot graphs along the graph period using multiplot and save the result into a .png file
    if files_column is None:
        files_column = 'all'
        df[files_column] = 'the source'
    for file_id in df[files_column].unique():
        # Selecting data and series to plot
        df_plot = df.query(f'{files_column} == @file_id')
        series_to_plot = {}
        if series_column is None:
            series_column = 'data'
            df_plot[series_column] = value_column
        for series in df_plot[series_column].unique():
            series_to_plot[series] = df_plot.query(f'{series_column} == @series')
        # Choosing axes and general plot parameters
        graph_times = eval(f'df_plot.index.{t_graph}.unique()')
        n_graphs = len(graph_times)
        if n_graphs == 1:
            axes = [1, 1]
        elif isinstance(axes[1], int) and n_graphs <= axes[1]:
            axes = [1, n_graphs]
        elif axes == 'default':
            axes = [math.ceil(n_graphs/4), 4]
        else:
            axes[0] = math.ceil(n_graphs/axes[1])
        x_col = t_unit
        y_col = value_column
        graph_id = {t_graph: graph_times}
        suptitle = f'Evolution of {value_column} by {t_unit} at {t_graph} scale over {t_global} for all {series_column} in {file_id}'
        # Choosing whether to save plots
        save_folder_cat = save_folder
        if save_folder is not None and files_column != 'all':
            save_folder_cat = save_folder_cat.replace('.png', f'_{file_id}.png')
        # Drawing plots
        multiplot(series_to_plot = series_to_plot, graph_id = graph_id, x_col = x_col, y_col = y_col, suptitle = suptitle, axes = axes, 
                  figsize = figsize, colors = colors, loc = loc, save_folder = save_folder_cat)
    del df
 
# Seasonal decomposition function with plotting option
# Compute the seasonal decomposition of one or several series
# If required, uses the multiplot function above to draw that seasonal decomposition at various timescales
# The timescales are: day, week, month, season, year
# def seasonal_plot(data, value_column, series_column, suptitle, axes,
#                   figsize = (18, 22), colors = 'random', save_folder = None):





