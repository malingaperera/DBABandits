import configparser
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame

import constants


def plot_histogram(statistics_dict, title, experiment_id):
    """
    Simple plot function to plot the average reward, and a line to show the best possible reward

    :param statistics_dict: list of statistic histograms
    :param title: title of the plot
    :param experiment_id: id of the current experiment
    """
    for statistic_name, statistic_histogram in statistics_dict.items():
        plt.plot(statistic_histogram, label=statistic_name)
    plt.title(title)
    plt.xlabel('Rounds')
    plt.legend()
    plt.savefig(get_experiment_folder_path(experiment_id) + title + '.png')
    plt.show()


def plot_histogram_v2(statistics_dict, window_size,  title, experiment_id):
    """
    Simple plot function to plot the average reward, and a line to show the best possible reward

    :param statistics_dict: list of statistic histograms
    :param window_size: size of the moving window
    :param title: title of the plot
    :param experiment_id: id of the current experiment
    """
    crop_amount = int(np.ceil(window_size/2))
    for statistic_name, statistic_histogram in statistics_dict.items():
        plt.plot(statistic_histogram[crop_amount:-crop_amount], label=statistic_name)
    plt.title(title)
    plt.xlabel('Rounds')
    plt.legend()
    plt.savefig(get_experiment_folder_path(experiment_id) + title + '.png')
    plt.show()


def get_experiment_folder_path(experiment_id):
    """
    Get the folder location of the experiment
    :param experiment_id: name of the experiment
    :return: file path as string
    """
    experiment_folder_path = constants.ROOT_DIR + constants.EXPERIMENT_FOLDER + '\\' + experiment_id + '\\'
    if not os.path.exists(experiment_folder_path):
        os.makedirs(experiment_folder_path)
    return experiment_folder_path


def get_workload_folder_path(experiment_id):
    """
    Get the folder location of the experiment
    :param experiment_id: name of the experiment
    :return: file path as string
    """
    experiment_folder_path = constants.ROOT_DIR + constants.WORKLOADS_FOLDER + '\\' + experiment_id + '\\'
    if not os.path.exists(experiment_folder_path):
        os.makedirs(experiment_folder_path)
    return experiment_folder_path


def plot_histogram_avg(statistic_dict, title, experiment_id):
    """
    Simple plot function to plot the average reward, and a line to show the best possible reward
    :param statistic_dict: list of statistic histograms
    :param title: title of the plot
    :param experiment_id: id of the current experiment
    """
    for statistic_name, statistic_list in statistic_dict.items():
        for i in range(1, len(statistic_list)):
            statistic_list[i] = statistic_list[i - 1] + statistic_list[i]
            statistic_list[i - 1] = statistic_list[i - 1] / i
        statistic_list[len(statistic_list) - 1] = statistic_list[len(statistic_list) - 1] / len(statistic_list)

    plot_histogram(statistic_dict, title, experiment_id)


def plot_moving_average(statistic_dict, window_size, title, experiment_id):
    """
    Simple plot function to plot the moving average of a histograms
    :param statistic_dict: list of statistic histograms
    :param window_size: size of the moving window
    :param title: title of the plot
    :param experiment_id: id of the current experiment
    """
    statistic_avg_dict = {}
    for statistic_name, statistic_list in statistic_dict.items():
        avg_mask = np.ones(window_size) / window_size
        statistic_list_avg = np.convolve(statistic_list, avg_mask, 'same')
        statistic_avg_dict[statistic_name] = statistic_list_avg

    plot_histogram_v2(statistic_avg_dict, window_size, title, experiment_id)


def get_queries_v2():
    """
    Read all the queries in the queries pointed by the QUERY_DICT_FILE constant
    :return: list of queries
    """
    # Reading the configuration for given experiment ID
    exp_config = configparser.ConfigParser()
    exp_config.read(constants.ROOT_DIR + constants.EXPERIMENT_CONFIG)

    # experiment id for the current run
    experiment_id = exp_config['general']['run_experiment']
    workload_file = str(exp_config[experiment_id]['workload_file'])

    queries = []
    with open(constants.ROOT_DIR + workload_file) as f:
        line = f.readline()
        while line:
            queries.append(json.loads(line))
            line = f.readline()
    return queries


def get_normalized(value, assumed_min, assumed_max, history_list):
    """
    This method gives a normalized reward based on the reward history

    :param value: current reward that we need to normalize
    :param history_list: rewards we got up to now, including the current reward
    :param assumed_min: assumed minimum value
    :param assumed_max: assumed maximum value
    :return: normalized reward (0 - 1)
    """
    if len(history_list) > 5:
        real_min = min(history_list) - 1
        real_max = max(history_list)
    else:
        real_min = min(min(history_list), assumed_min)
        real_max = max(max(history_list), assumed_max)
    return (value - real_min) / (real_max - real_min)


def update_dict_list(current, new):
    """
    This function does merging operation of 2 dictionaries with lists as values. This method adds only new values found
    in the new list to the old list

    :param current: current list
    :param new: new list
    :return: merged list
    """
    for table, predicates in new.items():
        if table not in current:
            current[table] = predicates
        else:
            temp_set = set(new[table]) - set(current[table])
            current[table] = current[table] + list(temp_set)
    return current


def plot_exp_report(exp_id, exp_report_list, measurement_names, log_y=False):
    """
    Creates a plot for several experiment reports
    :param exp_id: ID of the experiment
    :param exp_report_list: This can contain several exp report objects
    :param measurement_names: What measurement that we will use for y
    :param log_y: draw y axis in log scale
    """
    for measurement_name in measurement_names:
        comps = []
        final_df = DataFrame()
        for exp_report in exp_report_list:
            df = exp_report.data
            df[constants.DF_COL_COMP_ID] = exp_report.component_id
            final_df = pd.concat([final_df, df])
            comps.append(exp_report.component_id)

        final_df = final_df[final_df[constants.DF_COL_MEASURE_NAME] == measurement_name]
        # Error style = 'band' / 'bars'
        sns_plot = sns.relplot(x=constants.DF_COL_BATCH, y=constants.DF_COL_MEASURE_VALUE, hue=constants.DF_COL_COMP_ID,
                               kind="line", ci="sd", data=final_df, err_style="band")
        if log_y:
            sns_plot.set(yscale="log")
        plot_title = measurement_name + " Comparison"
        sns_plot.set(xlabel=constants.DF_COL_BATCH, ylabel=measurement_name)
        sns_plot.savefig(get_experiment_folder_path(exp_id) + plot_title + '.png')


def create_comparison_tables(exp_id, exp_report_list):
    """
    Create a CSV with numbers that are important for the comparison

    :param exp_id: ID of the experiment
    :param exp_report_list: This can contain several exp report objects
    :return:
    """
    final_df = DataFrame(
        columns=[constants.DF_COL_COMP_ID, constants.DF_COL_BATCH_COUNT, constants.MEASURE_HYP_BATCH_TIME,
                 constants.MEASURE_INDEX_RECOMMENDATION_COST, constants.MEASURE_INDEX_CREATION_COST,
                 constants.MEASURE_QUERY_EXECUTION_COST, constants.MEASURE_TOTAL_WORKLOAD_TIME])

    for exp_report in exp_report_list:
        data = exp_report.data
        component = exp_report.component_id
        rounds = exp_report.batches_per_rep
        reps = exp_report.reps

        # Get information from the data frame
        hyp_batch_time = get_avg_measure_value(data, constants.MEASURE_HYP_BATCH_TIME, reps)
        recommend_time = get_avg_measure_value(data, constants.MEASURE_INDEX_RECOMMENDATION_COST, reps)
        creation_time = get_avg_measure_value(data, constants.MEASURE_INDEX_CREATION_COST, reps)
        elapsed_time = get_avg_measure_value(data, constants.MEASURE_QUERY_EXECUTION_COST, reps)
        total_workload_time = get_avg_measure_value(data, constants.MEASURE_BATCH_TIME, reps) + hyp_batch_time

        # Adding to the final data frame
        final_df.loc[len(final_df)] = [component, rounds, hyp_batch_time, recommend_time, creation_time, elapsed_time,
                                       total_workload_time]

    final_df.round(4).to_csv(get_experiment_folder_path(exp_id) + 'comparison_table.csv')


# todo - remove min and max
def get_avg_measure_value(data, measure_name, reps):
    return (data[data[constants.DF_COL_MEASURE_NAME] == measure_name][constants.DF_COL_MEASURE_VALUE].sum())/reps


def get_sum_measure_value(data, measure_name):
    return data[data[constants.DF_COL_MEASURE_NAME] == measure_name][constants.DF_COL_MEASURE_VALUE].sum()


def change_experiment(exp_id):
    """
    Programmatically change the experiment

    :param exp_id: id of the new experiment
    """
    exp_config = configparser.ConfigParser()
    exp_config.read(constants.ROOT_DIR + constants.EXPERIMENT_CONFIG)
    exp_config['general']['run_experiment'] = exp_id
    with open(constants.ROOT_DIR + constants.EXPERIMENT_CONFIG, 'w') as configfile:
        exp_config.write(configfile)


def log_configs(logging, module):
    for variable in dir(module):
        if not variable.startswith('__'):
            logging.info(str(variable) + ': ' + str(getattr(module, variable)))
