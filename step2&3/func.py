import pickle
from time import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, HuberRegressor
from sklearn.metrics import *
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
import statsmodels.api as sm
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from tqdm.notebook import tqdm


def load_data(path):
    with open(path, 'rb') as file:
        unpicker = pickle.Unpickler(file, encoding='latin1')
        return unpicker.load()


def extract_outcome(data, xcolumn, input_column_post, input_column_pre=''):
    '''
    This function to extract outcome variable from data, return a pandas Series Object with name 'outcome',
    and exclude data that are both zero in pre-treatment and post-treatment period
    :param data: data to be processed
    :param xcolumn: name of vars of x
    :param input_column_post: name of outcomes in post-treatment period
    :param input_column_pre: name of outcomes in pre-treatment period
    :return: filtered outcome
    '''
    if input_column_pre != '':
        pre_sum = data[input_column_pre].sum(axis=1)
        post_sum = data[input_column_post].sum(axis=1)
        outcome = post_sum - pre_sum
        idx = ~((pre_sum == 0) & (post_sum == 0))
        re_dataframe = pd.concat([data[xcolumn], outcome.rename('outcome')], axis=1)
        return re_dataframe[idx].reset_index(drop=True)
    else:
        outcome = data[input_column_post].sum(axis=1)
        re_dataframe = pd.concat([data[xcolumn], outcome.rename('outcome')], axis=1)
        return re_dataframe.reset_index(drop=True)


def bound_data(data, bound_val, ifabs=False):
    '''
    This function returns data that are excluded from outliers value set by 'bound_val' parameter. The outcome column will be not greater than bound
    :param data: data to be processed
    :param bound_val: close interval
    :param ifabs: If bound applied to absolute value of the outcome
    :return: Data without outcome (absolute) value that greater than bound value
    '''
    if ifabs:
        idx = abs(data['outcome']) <= bound_val
    else:
        idx = data['outcome'] <= bound_val

    return data[idx].reset_index(drop=True)


def sample_data(data, sample_method='balanced', balance_bound='default', ifratio=True):
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
    assert (data['outcome'].max() - data['outcome'].min()) < 1000
    output_idx_mask = []

    random_seed = int(time()) % 100 * (int(time() * 10) % 10)
    outcome_vals = data['outcome']
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


def modeldata(data, model_name='xgb', logistic_cutoff=0, selected_xvars=[], ifplot=False,
              inputyvarname=''):
    '''
    Model the data by the parameters, return the prediction value
    :param data: data to be used to train and test on
    :param model_name: which model to use
    :param logistic_cutoff: the cut off of data when performing logistic regression
    :param selected_xvars: name of vars of train_x
    :param ifplot: whether to plot the results
    :param inputyvarname: name of label (y)
    :return: result and statistics of trained model
    '''
    random_seed = int(time()) % 100 * (int(time() * 10) % 10)
    # Extract x
    if selected_xvars == []:
        xnames = [x for x in data.columns if x != 'outcome']
        X_values = data.loc[:, xnames]
    else:
        X_values = data.loc[:, selected_xvars]
        xnames = selected_xvars
    Y_values = data['outcome']
    # Split train, test
    x_train, x_test, y_train, y_test = train_test_split(X_values, Y_values, test_size=0.25, random_state=random_seed)

    # Perform XGBoost
    if model_name == 'xgb':
        model = XGBRegressor()
        model.fit(x_train, y_train)
        test_pred = model.predict(x_test)
        mae_score = mean_absolute_error(y_test, test_pred)
        r2_val = r2_score(y_test, test_pred)
        coef_table = pd.DataFrame(data={'Coefficient': (model.feature_importances_).flatten()}, index=list(xnames))
        if ifplot:
            plt.figure()
            plt.xlim(-50, 50)
            plt.ylim(-50, 50)
            plt.scatter(test_pred, y_test)
            plt.title(inputyvarname)
            plt.xlabel('Predicting Values')
            plt.ylabel('Real Values')
            plt.show()
        return model.predict(X_values), mae_score, r2_val, coef_table
    # Perform Random Forest
    elif model_name == 'randomforest':
        model = RandomForestRegressor()
        model.fit(x_train, y_train)
        test_pred = model.predict(x_test)
        mae_score = mean_absolute_error(y_test, test_pred)
        r2_val = r2_score(y_test, test_pred)
        return model.predict(X_values), mae_score, r2_val
    # Perform Linear Regression
    elif model_name == 'linear':
        model = LinearRegression()
        model.fit(x_train, y_train)
        test_pred = model.predict(x_test)
        mae_score = mean_absolute_error(y_test, test_pred)
        r2_val = r2_score(y_test, test_pred)
        # Get the coefficients of the model
        coef_table = pd.DataFrame(data={'Coefficient': (model.coef_).flatten()}, index=list(xnames))
        if ifplot:
            plt.xlim(-30, 30)
            plt.ylim(-30, 30)
            plt.figure()
            plt.scatter(test_pred, y_test)
            plt.title(inputyvarname)
            plt.xlabel('Predicting Values')
            plt.ylabel('Real Values')
            plt.show()
        return model.predict(X_values), mae_score, r2_val, coef_table
    # Perform Logistic Regression
    elif model_name == 'logistics':
        model_log = LogisticRegression(random_state=random_seed + 7, class_weight='balanced', max_iter=300)
        model_log.fit(x_train, y_train > logistic_cutoff)
        y_1d_proba = model_log.predict_proba(x_test)
        fpr, tpr, threshold = roc_curve(y_test > logistic_cutoff, y_1d_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        y_pred_test = y_1d_proba[:, 1] > threshold[optimal_idx]
        coef_table = pd.DataFrame(data={'Coefficient': (model_log.coef_).flatten()}, index=list(xnames))
        return model_log.predict(X_values), f1_score(y_test > logistic_cutoff, y_pred_test), roc_auc, coef_table
