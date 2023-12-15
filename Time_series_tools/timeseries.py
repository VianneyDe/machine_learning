from general_utilities import *
from seriessummary import *
from seriestests import *
## !!! Don't go over 100 columns in .py file
## TODO: Have a repository for each step/stage in the process
## TODO: Add a conclusion attribute to the SeriesTests class (dataframe summing up the results)
## TODO: Add all the auto-correlation features mentioned by Hyndman (use the window_size argument in tsfel?)
## as well as partial autocorrelations
## TODO: Add the usual decomposition as attributes to the time series, similar to summary
## TODO: Add the STL features to summary classes
## TODO: Add the following features (if not already provided by TSFEL):
#       2. via scipy.stats:
#               a. Guerrero method (optimal lambda for Box-Cox transformation)
#               b. shift_kl_max and shift_kl_index (using scipy.stats.entropy and a window size)
#       2. by computing tsfel features for a well chosen series ? :
#               a. n_crossing_points (using zero_crossing_points)
#               b. var_tiled_mean ('stability') using var, mean and a window size
#               c. var_tiled_var ('lumpiness') using var, mean and a window size
#               d. shift_level_max and shift_level_index (using mean absolute diff and window size?)
#               e. shift_var_max and shift_var_index (same as above on the series of variances?)
#               f. longest flat spot (peak-to-peal distance?)
#        4. Hurst coefficient (with hurst package?)
class TimeSeries():
    """ Main class containing one or several time series, time index and filename information"""
    def __init__(self, filename = None, separator = ',', encoding = 'utf-8', 
                 series_data = None, date_column = 'index', 
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
            print('Generating hourly random uniform time series on [0, 1[ over the year 2022')
            self.series_data = generate_random_time_series(start = '2022-01-01', end = '2022-12-31')
        print(f'Creating time index')
        self.series_data = create_time_index(self.series_data, date_column = date_column)
        self.series_values_columns = series_values_columns
        print('Identifying value columns for the time series')
        if series_values_columns == 'default':
            all_columns = self.series_data.columns
            print(f'Columns: {list(all_columns)}')
            value_columns = [col for col in all_columns if is_numeric_dtype(self.series_data[col])]
            self.series_values_columns = value_columns
        print('Creating time series summary')
        self.summary = SeriesSummary(full_data = self.series_data,
                                     value_columns = self.series_values_columns, 
                                     id_columns = series_names_columns)
        print('Running standard tests through the data')
        self.tests = SeriesTests(full_data = self.series_data,
                                     value_columns = self.series_values_columns, 
                                     id_columns = series_names_columns)
    def save_data(self):
        self.series_data.to_csv(self.file['name'], sep = self.file['separator'], encoding = self.file['encoding'])
        self.file.saved = True
    
    # @property
    # def series_summary(self):
    #     return self.series_summary
    
    # @series_summary.setter
    # def features(self):