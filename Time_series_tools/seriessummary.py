from general_utilities import *
class SeriesSummary:
    """Main summary class containing the names value_columns of relevant columns, 
    the names id_columns of the eventual id column for each of those columns 
    and the number of different ids in each of the id_columns. 
    Can also contain temporal, spectral and statistical domain features (if required by user)
    Argument value_columns: a list of column names. 
    Argument id_columns: a dict with: - the elements of value_columns as keys
                                      - the name of the corresponding id column as value for each key
    """
    def __init__(self, full_data, value_columns, id_columns = None,
                 temporal = True, spectral = True, statistical = True):
        if id_columns is None:
            self.general = pd.DataFrame({'value_column': value_columns, 
                                         'id_column': ['None']*len(value_columns), 
                                         'number_of_series': [1]*len(value_columns)})
        else:
            self.general = pd.DataFrame({'value_columns': value_columns})
            self.general['id_column'] = [id_columns[key] 
                                                for key in self.general['value_columns']]
            self.general['series_quantity'] = [len(full_data[id].unique())
                                                for id in self.general['id_column']]
        if temporal:
            print('Extracing temporal domain features')
            features_summary_file = tsfel.get_features_by_domain("temporal")
            features = tsfel.time_series_features_extractor(features_summary_file, full_data[value_columns])
            self.temporal = features.rename(columns = dict(zip(features.columns, 
                                            [col.replace('0_', '') for col in features.columns])))
        if spectral:
            print('Extracing spectral domain features')
            features_summary_file = tsfel.get_features_by_domain("spectral")
            del features_summary_file['spectral']['Spectral roll-off']
            del features_summary_file['spectral']['Spectral roll-on']
            del features_summary_file['spectral']['Power bandwidth']
            features = tsfel.time_series_features_extractor(features_summary_file, full_data[value_columns])
            self.spectral = features.rename(columns = dict(zip(features.columns, 
                                            [col.replace('0_', '') for col in features.columns])))
        if statistical:
            print('Extracing statistical domain features')
            features_summary_file = tsfel.get_features_by_domain("statistical")
            features = tsfel.time_series_features_extractor(features_summary_file, full_data[value_columns])
            self.statistical = features.rename(columns = dict(zip(features.columns, 
                                            [col.replace('0_', '') for col in features.columns])))
            

