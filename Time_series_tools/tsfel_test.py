# %%)))
import tsfel
import zipfile
import numpy as np
import pandas as pd

# Load the dataset from online repository
# !wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip

# # Unzip the dataset
# zip_ref = zipfile.ZipFile("UCI HAR Dataset.zip", 'r')
# zip_ref.extractall()
# zip_ref.close()

# Store the dataset as a Pandas dataframe.
x_train_sig = np.loadtxt('UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt', dtype='float32')
X_train_sig = pd.DataFrame(np.hstack(x_train_sig), columns=["total_acc_x"])

cfg_file = tsfel.get_features_by_domain()                                                        # If no argument is passed retrieves all available features
X_train = tsfel.time_series_features_extractor(cfg_file, X_train_sig, fs=50, window_size=250)    # Receives a time series sampled at 50 Hz, divides into windows of size 250 (i.e. 5 seconds) and extracts all features

# %% )
X_train.to_csv('feature_calc.csv')
# %%
