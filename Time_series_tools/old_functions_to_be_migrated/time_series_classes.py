import pdb
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from datetime import date, datetime
import tsfel

# Random color generation function
def random_colors(n):
    colors = []
    for color in range(n):
        r = np.round(np.random.rand(), 1)
        g = np.round(np.random.rand(), 1)
        b = np.round(np.random.rand(), 1)
    colors.append([r, g, b])
    return colors

# Function for converting a column or index of a dataframe into a time index
def create_time_index(data, date_column = 'index', timezone = 'default'):
    if date_column == 'index':
        print('Using original index as time index')
    else:
        print(f'Using {date_column} as new time index')
        data = data.set_index(date_column, drop = True)
    try:
        utc = (data.index.tzinfo is not None)
    except AttributeError:
        print('ERROR: Index cannot be converted to time index, returning data with nontime index')
    else:
        data.index = pd.to_datetime(data.index, utc = utc)
        if utc and timezone != 'default':
            data.index = data.index.tz_convert(timezone)
    finally:
        return data

# Function for generating random time series
def generate_random_time_series(start, end, periods = None, freq = 'H', tz = 'UTC', min = 0, max = 1):
    rng_series = pd.DataFrame([])
    rng_series['date'] = pd.date_range(start = start, end = end, periods = periods, freq = freq, tz = tz)
    rng_series = rng_series.set_index('date', drop = True)
    rng = np.random.default_rng()
    rng_series['value'] = min + (max - min)*rng.random(rng_series.shape[0])
    return rng_series

# class PlotSeries:
#     """ Class to enable seamless plotting of multiple graphs at once"""
#     def __init__(self, curves_var_dict = None, graphs_dict = None, general_params = 'default', 
#                  single_graph_params = 'default',save_folder = None):
#         if general_params == 'default':
#             n_graphs = len(graphs_dict.keys())

## TODO: Make file into a class in itself?
## TODO: Make series_summary into a class in itself?
## TODO: Add summary subclass: spectral_summary? statistical_summary? 
## TODO: Make one file for each class
## !!! Don't go over 100 columns in .py file
## TODO: Have a repository for eahc step/stage in the process

class TimeSeries():
    """ Main class containing one or several time series, time index and filename information"""
    def __init__(self, filename = None, separator = ',', encoding = 'utf-8', series_data = None, date_column = 'index', 
                 series_values_columns = 'default', series_names_columns = None):
        print('Fetching time series data')
        self.file = {'name': filename, 'separator': separator, 'encoding': encoding, 'saved': False}
        if filename is not None:
            print(f'Loading dataset from {filename}')
            try:
                self.series_data = pd.read_csv(filename, sep = separator, encoding = encoding)
            except FileNotFoundError:
                print('ERROR: File not found')
                print('Using dataset from input argument')
                self.series_data = series_data
            else:
                self.file['saved'] = True
        elif series_data is not None:
            print('Using dataset from input argument')
            self.series_data = series_data
        else:
            print('Generating hourly random uniform time series on [0, 1[ for all hours in year 2022')
            self.series_data = generate_random_time_series(start = '2022-01-01', end = '2022-12-31')
        print(f'Creating time index')
        self.series_data = create_time_index(self.series_data, date_column = date_column)
        self.series_values_columns = series_values_columns
        print('Identifying value columns for the time series')
        if series_values_columns == 'default':
            all_columns = self.series_data.columns
            print(f'Columns: {all_columns}')
            value_columns = [col for col in all_columns if is_numeric_dtype(col)]
        print('Creating time series summary')
        if series_names_columns is None:
            self.series_summary = pd.DataFrame({'series_values_columns': series_values_columns, 
                                               'series_names_columns': ['None']*len(value_columns), 
                                               'series_quantity': [1]*len(value_columns)})
        else:
            self.series_summary = pd.DataFrame({'series_values_columns': series_values_columns})
            self.series_summary['series_id'] = [series_names_columns[key] for key in self.series_summary['series_values_columns']]
        
    def save_data(self):
        self.series_data.to_csv(self.file['name'], sep = self.file['separator'], encoding = self.file['encoding'])
        self.file.saved = True
    
    # @property
    # def series_summary(self):
    #     return self.series_summary
    
    # @series_summary.setter
    # def features(self):

# class TimePlotSeries(TimeSeries, PlotSeries):
#     """TimeSeries subclass with additional attributes for easier plotting"""
#     def __init__(self, filename=None, separator=',', encoding='utf-8', series_data=None, date_column='index', 
#                  series_values_columns='default', series_names_columns=None):
#         super().__init__(filename, separator, encoding, series_data, date_column, series_values_columns,
#                           series_names_columns)
        