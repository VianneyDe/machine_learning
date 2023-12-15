import pdb
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from datetime import date, datetime
import tsfel
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import kpss, adfuller
from arch.unitroot import ADF, DFGLS, PhillipsPerron, KPSS, ZivotAndrews
from pmdarima.arima import ndiffs, nsdiffs

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