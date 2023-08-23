import pickle
from time import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.markers as markers
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, HuberRegressor
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

all_markers = markers.MarkerStyle.markers
marker_list = list(all_markers.keys())


def load_data(path):
    with open(path, 'rb') as file:
        unpicker = pickle.Unpickler(file, encoding='latin1')
        return unpicker.load()


def sample_data(data, sample_method='balanced', balance_bound='default', ifratio=True, keyword='outcome'):
    '''
    This method will tried to balanced the extreme skewed data, there are three ways of sampling:
    This method should applied after excluding the extreme values. Data should be in reasonable range.
    :param data: data to be processed
    :param sample_method: default is 'balanced', this will force outcome value counts same at each value.
    'sqrt' will sample each counts(a/min) sqrt(a/min)*a, times,
    'log' will sample each counts(a/min) log(a/min)*a, times
    :param balance_bound: a user specified lower bound for sampling
    :param ifratio: default to be true, sampled data by the ratio of original data
    :return: data after sampled
    '''
    assert (data[keyword].max() - data[keyword].min()) < 1000
    output_idx_mask = []

    random_seed = int(time()) % 100 * (int(time() * 10) % 10)
    outcome_vals = data[keyword]
    index = np.arange(len(outcome_vals))  # Index of outcome values
    count_values = Counter(outcome_vals)  # Counts of outcome values
    minimalcounts = min(list(count_values.values()))  # min counts of outcome values
    # Using balance bound or minimal count as condition
    if balance_bound != 'default':
        multi_cons = balance_bound
    else:
        multi_cons = minimalcounts

    # Perform Square root sampling
    if sample_method == 'sqrt':
        for cur_outcome_value in list(set(outcome_vals)):
            cur_idx = index[outcome_vals == cur_outcome_value]
            # Get the count of current value
            cur_value_count = count_values[cur_outcome_value]
            if cur_value_count < multi_cons:
                output_idx_mask += cur_idx.tolist()
                continue
            # Sqrt downsampled under original ratio
            if ifratio:
                new_count = int(np.sqrt(cur_value_count / multi_cons) * multi_cons)
            # Sqrt downsampled
            else:
                new_count = int(np.sqrt(cur_value_count))
            # Sample data using new count
            reduced_idx = pd.Series(cur_idx).sample(n=new_count, random_state=(random_seed - cur_value_count) % 100,
                                                    replace=False)
            output_idx_mask += reduced_idx.tolist()

    # Perform log sampling
    elif sample_method == 'log':
        for cur_outcome_value in list(set(outcome_vals)):
            cur_idx = index[outcome_vals == cur_outcome_value]
            cur_value_count = count_values[cur_outcome_value]
            if cur_value_count < multi_cons:
                output_idx_mask += cur_idx.tolist()
                continue
            if ifratio:
                new_count = int(np.log(np.exp(1) * cur_value_count / multi_cons) * multi_cons)
            else:
                new_count = int(np.sqrt(cur_value_count))
            reduced_idx = pd.Series(cur_idx).sample(n=new_count, random_state=(random_seed - cur_value_count) % 100,
                                                    replace=False)
            output_idx_mask += reduced_idx.tolist()

    # Perform balanced sampling
    elif sample_method == 'balanced':
        for cur_outcome_value in list(set(outcome_vals)):
            cur_idx = index[outcome_vals == cur_outcome_value]
            cur_value_count = count_values[cur_outcome_value]
            if cur_value_count < multi_cons:
                output_idx_mask += cur_idx.tolist()
                continue
            new_count = multi_cons
            reduced_idx = pd.Series(cur_idx).sample(n=new_count, random_state=(random_seed - cur_outcome_value) % 100,
                                                    replace=False)
            output_idx_mask += reduced_idx.tolist()

    return data.iloc[output_idx_mask].reset_index(drop=True)

