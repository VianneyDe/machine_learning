import pandas as pd


u = pd.read_hdf("train_dc2020.h5", 'data')

u.head(200).to_csv('train_cfm.csv')