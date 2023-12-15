from general_utilities import *

class SeriesTests:
    """Class gathering standard statistical tests for a time series. Includes:
            - Tests for whether the series is white noise: Ljung-Box, Box-Pierce
            - Stationarity and unit root tests: ADF, KPSS, Phillips-Perron, DFGLS, Zivot-Andrews
            - Tests for the number of differentiation to reach stationarity: ADF, KPSS, PP
            - Tests for the number of differention of the seasonal component: CH, OCSB"""
    def __init__(self, full_data, value_columns, id_columns = None,):
        full_data = full_data[value_columns]
        print('Testing if the series is a white noise process')
        self.white_noise = acorr_ljungbox(full_data[value_columns], boxpierce = True)
        self.white_noise.index.name = 'lags'
        print('Testing if the series is stationary')
        self.adf = ADF(full_data)
        print('Testing is the series has an unit root')
        self.kpss = KPSS(full_data)
        self.dfgls = DFGLS(full_data)
        self.pp = PhillipsPerron(full_data)
        print('Testing if the series has an unit root with a single break point')
        self.za = ZivotAndrews(full_data)
        print('Determining the number of differentiations to reach stationarity')
        self.n_adf = ndiffs(full_data, test = 'adf')
        self.n_kpss = ndiffs(full_data, test = 'kpss')
        self.n_pp = ndiffs(full_data, test = 'pp')
        print('Determining the number of seasonal differentiations to reach stationarity')
        self.ch = nsdiffs(full_data, m = 24, test = 'ch')
        self.ocsb = nsdiffs(full_data, m = 24, test = 'ocsb')
        self.het = het_arch(full_data)
        # self.results = pd.DataFrame([])
        #self.results['test']
        #self.results['parameter'] ?
        #self.results['statistic']
        #self.results['null']
        #self.results['alternative']
        #self.results['p-value']
        #self.results['outcome']

