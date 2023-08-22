import pickle
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pandas as pd
from timeit import default_timer
from sklearn.cluster import KMeans
import seaborn as sns
from tqdm import tqdm
from scipy.stats import ttest_ind, ks_2samp
from sklearn.decomposition import PCA
import pandas as pd
import pyarrow.parquet as pq
import imgkit


# Store variables' name by category
def variable_list(in_key):
    # Hide internal variables name
    artist_related_variables = []
    # Hide internal variables name
    user_related_variables = []
    user_segmentation = [
        'light_listener',
        'moderate_listener',
        'super_listener'
    ]
    id_info = [
        'artist_gid',
        'user_id',
        'date',
    ]
    pre1names = [
        'saves_pre_period_1',
        'follows_pre_period_1',
        'playlists_pre_period_1',
        'tickets_pre_period_1',
        'merch_pre_period_1',
        'shares_pre_period_1',
        'cumsaves_pre_period_1',
        'cumfollows_pre_period_1',
        'cumplaylists_pre_period_1',
        'cumtickets_pre_period_1',
        'cummerch_pre_period_1',
        'cumshares_pre_period_1',
        'streams_active_streams_pre_period_1',
        'streams_programmed_streams_pre_period_1'
    ]
    pre2names = [
        'saves_pre_period_2',
        'follows_pre_period_2',
        'playlists_pre_period_2',
        'tickets_pre_period_2',
        'merch_pre_period_2',
        'shares_pre_period_2',
        'cumsaves_pre_period_2',
        'cumfollows_pre_period_2',
        'cumplaylists_pre_period_2',
        'cumtickets_pre_period_2',
        'cummerch_pre_period_2',
        'cumshares_pre_period_2',
        'streams_active_streams_pre_period_2',
        'streams_programmed_streams_pre_period_2'
    ]
    pre3names = [
        'saves_pre_period_3',
        'follows_pre_period_3',
        'playlists_pre_period_3',
        'tickets_pre_period_3',
        'merch_pre_period_3',
        'shares_pre_period_3',
        'cumsaves_pre_period_3',
        'cumfollows_pre_period_3',
        'cumplaylists_pre_period_3',
        'cumtickets_pre_period_3',
        'cummerch_pre_period_3',
        'cumshares_pre_period_3',
        'streams_active_streams_pre_period_3',
        'streams_programmed_streams_pre_period_3'
    ]
    post2wknames = [
        'saves_following_two_weeks',
        'follows_following_two_weeks',
        'playlists_following_two_weeks',
        'tickets_following_two_weeks',
        'merch_following_two_weeks',
        'shares_following_two_weeks',
        'streams_active_streams_following_two_weeks',
        'streams_programmed_streams_following_two_weeks'
    ]
    post4wknames = [
        'saves_following_four_weeks',
        'follows_following_four_weeks',
        'playlists_following_four_weeks',
        'tickets_following_four_weeks',
        'merch_following_four_weeks',
        'shares_following_four_weeks',
        'streams_active_streams_following_four_weeks',
        'streams_programmed_streams_following_four_weeks'
    ]
    treatment_act = [
        'saves_treatment_period',
        'follows_treatment_period',
        'playlists_treatment_period',
        'tickets_treatment_period',
        'merch_treatment_period',
        'shares_treatment_period',
        'streams_active_streams_treatment_period',
        'streams_programmed_streams_treatment_period'
    ]
    treatment_cum = [
        'cumsaves_treatment_period',
        'cumfollows_treatment_period',
        'cumplaylists_treatment_period',
        'cumtickets_treatment_period',
        'cummerch_treatment_period',
    ]
    var_name_dict = {
        'art': artist_related_variables,
        'usr': user_related_variables,
        'seg': user_segmentation,
        'pre1': pre1names,
        'pre2': pre2names,
        'pre3': pre3names,
        'post2': post2wknames,
        'post4': post4wknames,
        'treat_act': treatment_act,
        'treat_cum': treatment_cum
    }
    return var_name_dict[in_key]


# Load data
def load_data(path):
    '''
    :param path: the relative path to the current directory
    :return: data of dataframe type
    '''
    with open(path, 'rb') as file:
        unpicker = pickle.Unpickler(file, encoding='latin1')
        return unpicker.load()


# Summarize outcome given input data
def outcomes_summary(outcome_var_pre, outcome_var_post, input_data, iftwo=False):
    '''
    :param outcome_var_pre: variable name of outcomes in pre-treatment period
    :param outcome_var_post: variable name of outcomes in post-treatment period
    :param input_data: the outcomes that needs to be summarized
    :param iftwo: default false, set this to True if outcome_var_pre or outcome_var_post
     is combined variable (outcome_var_pre is a list of two).
    :return: statistics of outcomes
    '''
    if type(outcome_var_pre) == list:
        iftwo = True
    if not iftwo:
        pre_outcome = input_data[outcome_var_pre]
        post_outcome = input_data[outcome_var_post]

    else:
        pre_outcome = input_data[outcome_var_post[0]] + input_data[outcome_var_post[1]]
        post_outcome = input_data[outcome_var_pre[0]] + input_data[outcome_var_pre[1]]

    changes = post_outcome - pre_outcome
    bothzero_idx = (pre_outcome == 0) & (post_outcome == 0)
    num_bothzero = np.sum(bothzero_idx)

    too_big_values_idx = np.abs(changes) > 10 * np.std(changes)
    toobigval_counts = np.sum(too_big_values_idx)

    print('Data Shape', input_data.shape)
    if iftwo:
        print(
            'For the outcome: ' + outcome_var_post[0] + ', ' + outcome_var_pre[0] + ' \n' + outcome_var_post[1] + ' ,' +
            outcome_var_pre[1])

        print("num of rows in post/pre are all zeros: ", num_bothzero)
        print("num of rows in post/pre are not all zeros: ", input_data.shape[0] - num_bothzero)
        print("num of rows with absolute changes greater than 10x std: ", toobigval_counts)

    else:
        print('For the outcome: ' + outcome_var_post + ',' + outcome_var_pre)
        print("num of rows in post/pre are all zeros: ", num_bothzero)
        print("num of rows in post/pre are not all zeros: ", input_data.shape[0] - num_bothzero)
        print("num of rows with absolute changes greater than 10x std: ", toobigval_counts)
    print()


# Clean data exclude 0s and 20x std values
def clean_data(outcome_var_pre, outcome_var_post, input_data, iftwo=False):
    '''
    :param outcome_var_pre: variable name of outcomes in pre-treatment period
    :param outcome_var_post: variable name of outcomes in post-treatment period
    :param input_data: data to be processed
    :param iftwo: default false, set this to True if outcome_var_pre or outcome_var_post
     is combined variable (outcome_var_pre is a list of two).
    :return: cleaned data
    '''
    if type(outcome_var_pre) == list:
        iftwo = True
    std_limit = 20

    if not iftwo:
        pre_outcome = input_data[outcome_var_pre]
        post_outcome = input_data[outcome_var_post]
    else:
        pre_outcome = input_data[outcome_var_post[0]] + input_data[outcome_var_post[1]]
        post_outcome = input_data[outcome_var_pre[0]] + input_data[outcome_var_pre[1]]

    bothzero_idx = (pre_outcome == 0) & (post_outcome == 0)
    changes = post_outcome - pre_outcome

    num_bothzero = np.sum(bothzero_idx)

    too_big_values_idx = np.abs(changes) > std_limit * changes.std()
    toobigval_counts = np.sum(too_big_values_idx)

    print("num of rows in post/pre are both zeros: ", num_bothzero)
    print("num of rows in post/pre are at least one non-zeros: ", input_data.shape[0] - num_bothzero)
    print("num of rows with absolute changes greater than 20x std: ", toobigval_counts)
    cleaned_data = input_data[~(bothzero_idx | too_big_values_idx)].dropna()

    print("The size of cleaned data : ", cleaned_data.shape)
    return cleaned_data


# Perform k-means clustering
def kmeans_confoudners(data, num_clus, confoudners_input):
    '''
    :param data: data to be clustered
    :param num_clus: number of cluster
    :param confoudners_input: name of the confounders
    :return: label of clusters that each data that belongs
    '''
    confounders_data = data[confoudners_input]
    zero_pos = confounders_data.max() - confounders_data.min() == 0
    scaled_confounder_data = (confounders_data - confounders_data.min()) / (
            confounders_data.max() - confounders_data.min())
    scaled_confounder_data.loc[:, zero_pos] = confounders_data.loc[:, zero_pos]
    km = KMeans(n_clusters=num_clus, n_init=10, random_state=0)
    km.fit(scaled_confounder_data)
    return km.labels_


# Fetch case/control label by group
def get_case_control_label(outcome_var_pre, outcome_var_post, input_data, control_up, case_down, iftwo=False):
    '''
    :param outcome_var_pre: the name of pre-period variable
    :param outcome_var_post: the name of post-period variable
    :param input_data: input dataset(pandas.dataframe)
    :param control_up: Upper limit of control group
    :param case_down: Lower limit of case group
    :param iftwo: default false. If you need to input combined variable as outcome ( outcome_var_pre is a list of two). Set this to True
    :return:
    '''
    if type(outcome_var_pre) == list:
        iftwo = True
    if not iftwo:
        pre_outcome = input_data[outcome_var_pre]
        post_outcome = input_data[outcome_var_post]
    else:
        pre_outcome = input_data[outcome_var_post[0]] + input_data[outcome_var_post[1]]
        post_outcome = input_data[outcome_var_pre[0]] + input_data[outcome_var_pre[1]]

    changes = post_outcome - pre_outcome
    case_label = (changes >= case_down).tolist()
    control_label = (changes <= control_up).tolist()

    case_control_labels = [''] * len(changes)
    for idx in range(len(changes)):
        if case_label[idx]:
            case_control_labels[idx] = 'case'
        elif control_label[idx]:
            case_control_labels[idx] = 'control'

    return case_control_labels


# Calculte smd before and after matching
def smd_calculation_confounders(data, input_clus_label, input_confounder_label, input_case_label):
    '''
    :param data: the data being calculated
    :param input_clus_label: cluster label of each data
    :param input_confounder_label: list of name of confounders
    :param input_case_label: case
    :return:
    '''
    smd_after = {}
    smd_before = {}
    le_before = []
    le_after = []
    data['cluster_label'] = input_clus_label
    data['case_label'] = input_case_label
    var_confonders = input_confounder_label

    var_clus = []
    for clus in set(input_clus_label):
        cur_clus = np.array(input_case_label)[np.array(input_clus_label) == clus]
        num_case = np.sum(cur_clus == 'case')
        num_control = np.sum(cur_clus == 'control')
        if num_case > 1 and num_control > 1:
            var_clus.append(clus)

    # Calculate smd before matching
    for col in var_confonders:
        mean_data = data.loc[data['case_label'] == 'case', col]
        mean_case = mean_data.mean()
        control_data = data.loc[data['case_label'] == 'control', col]
        mean_control = control_data.mean()
        n1 = len(mean_data)
        n2 = len(control_data)
        std_pooled = np.sqrt(((n1 - 1) * mean_data.std() ** 2 + (n2 - 1) * control_data.std() ** 2) / (n1 + n2 - 2))
        if std_pooled == 0:
            std_pooled = 1
        smd_before[col] = abs(mean_case - mean_control) / std_pooled
        le_before.append(abs(mean_case - mean_control) / std_pooled)

    # Calculate smd after matching
    for clus in var_clus:
        cur_clus = data[data['cluster_label'] == clus]
        smd_clus = {}
        for col in var_confonders:
            mean_data = cur_clus.loc[cur_clus['case_label'] == 'case', col]
            control_data = cur_clus.loc[cur_clus['case_label'] == 'control', col]
            n1 = len(mean_data)
            n2 = len(control_data)
            mean_case = mean_data.mean()
            mean_control = control_data.mean()
            std_pooled = np.sqrt(((n1 - 1) * mean_data.std() ** 2 + (n2 - 1) * control_data.std() ** 2) / (n1 + n2 - 2))
            if std_pooled == 0:
                std_pooled = 1
            smd_clus[col] = abs(mean_case - mean_control) / std_pooled
            le_after.append(abs(mean_case - mean_control) / std_pooled)
        smd_after[clus] = smd_clus

    return smd_before, smd_after, le_before, le_after


# Plot SMD of case and control before and after clustering ('./plots' folder is required to save the output plots)
def smd_plot_calmean(input_before_smd, input_after_smd, outcome_var_pre, outcome_var_post, control_up,
                     case_up, iftwo=False, name_suffix=''):
    '''
    :param input_before_smd:
    :param input_after_smd:
    :param outcome_var_pre:
    :param outcome_var_post:
    :param control_up:
    :param case_up:
    :param iftwo:
    :param name_suffix:
    :return:
    '''
    if type(outcome_var_pre) == list:
        iftwo = True
    if iftwo:
        outcome_name = outcome_var_pre[0][:outcome_var_pre[0].find('_') - 1]
        prename = 'pre period ' + outcome_var_pre[0].split('_')[-1]
        postname = '_'.join(outcome_var_post[0].split('_')[-3:])
    else:
        outcome_name = outcome_var_pre[:outcome_var_pre.find('_') - 1]
        prename = 'pre period ' + outcome_var_pre.split('_')[-1]
        postname = '_'.join(outcome_var_post.split('_')[-3:])
    plt.figure()
    sns.kdeplot(data=input_before_smd, fill=True, common_norm=False, alpha=0.5, cut=0, label='Before')
    sns.kdeplot(data=input_after_smd, fill=True, common_norm=False, alpha=0.5, cut=0, label='After')
    plt.xlabel('SMD')
    plt.ylabel('Density')
    plt.title('SMD after matching')
    plt.legend(['Before', 'After'])
    plt.show()
    plt.savefig(
        f'''./plots/smdplot_{outcome_name}_{prename}_{postname}_case_{case_up}_control_{control_up}{name_suffix}.png''')


# Calculate the statistics of the dataset
def cal_stats(input_data, input_causes_list, input_clus_label, input_case_label):
    '''
    :param input_data: data to be processed
    :param input_causes_list: list of causes
    :param input_clus_label: label of cluster that each data belongs to
    :param input_case_label: label of data that belongs to case
    :return: tables of statistics
    '''
    input_data['cluster_label'] = input_clus_label
    input_data['case_label'] = input_case_label

    control_all_var = []
    case_all_var = []
    avg_pvalue_ttest = []
    avg_pvalue_kstest = []

    t_dict = {}
    ks_dict = {}
    co_dict = {}
    co_all_list = []

    t_all_list = []
    ks_all_list = []
    t_count_sig = []
    ks_count_sig = []
    var_clus = []

    for clus in set(input_clus_label):
        cur_clus = np.array(input_case_label)[np.array(input_clus_label) == clus]
        num_case = np.sum(cur_clus == 'case')
        num_control = np.sum(cur_clus == 'control')
        if num_case > 1 and num_control > 1:
            var_clus.append(clus)

    for var in input_causes_list:
        case_clus = []
        control_clus = []
        ttest = []
        kstest = []
        colist = []
        for clus in var_clus:
            # Prepare data
            cur_data = input_data[input_data['cluster_label'] == clus]
            case_values = cur_data[cur_data['case_label'] == 'case'][var]
            control_values = cur_data[cur_data['case_label'] == 'control'][var]
            case_avg = np.mean(case_values)
            control_avg = np.mean(control_values)

            # Calculate pooled std
            n1 = len(case_values)
            n2 = len(control_values)
            pooled_std = np.sqrt(
                ((n1 - 1) * case_values.std() ** 2 + (n2 - 1) * control_values.std() ** 2) / (n1 + n2 - 2))
            case_clus.append(case_avg)
            control_clus.append(control_avg)
            if len(case_values) == 0 or len(control_values) == 0:
                print(f"Empty data at cluster {clus}, variable {var}. Skipping.")
                continue

            # Perform KS test, t-test
            a, pks = ks_2samp(case_values, control_values)
            b, pts = ttest_ind(case_values, control_values)

            # Calculate Cohen's d
            cohensd = abs(case_avg - control_avg) / pooled_std
            ttest.append(pts)
            kstest.append(pks)
            colist.append(cohensd)

        t_all_list += (ttest)
        ks_all_list += (kstest)
        ks_dict[var] = kstest
        t_dict[var] = ttest
        co_dict[var] = colist

        case_all_var.append(np.mean(case_clus))
        control_all_var.append(np.mean(control_clus))
        avg_pvalue_ttest.append(np.mean(ttest))
        avg_pvalue_kstest.append(np.mean(kstest))
        co_all_list.append(np.mean(colist))
        t_count_sig.append(np.sum(np.array(ttest, dtype=float) < 0.05))
        ks_count_sig.append(np.sum(np.array(kstest, dtype=float) < 0.05))

    # Create result dataframe
    num_clus = len(list(set(input_clus_label)))
    Result_table = pd.DataFrame({
        'Case': case_all_var,
        'Control': control_all_var,
        'K-S test p value(sig. proportion)': np.array(ks_count_sig) / num_clus,
        'Cohens d': co_all_list,
        'Independent t-statistics p value(sig. proportion)': np.array(t_count_sig) / num_clus,
    }, index=input_causes_list)

    return Result_table
