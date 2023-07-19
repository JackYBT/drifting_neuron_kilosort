import numpy as np
from spike_raster_display import extract_single_neuron_spike_times_for_specific_trial, center_around_cue, sliding_histogram, extract_processed_neuron_raster
from raw_voltage import extractPeakChannel, getDataAndPlot, getMultipleTrialVoltage
from matplotlib import pyplot as plt
from scipy.signal import correlate
import os
import scipy.stats
import time
import math
from scipy.io import loadmat
import neo
import quantities as pq
from elephant.spike_train_correlation import spike_time_tiling_coefficient
from uni_multi_variate_helpers import *


def return_utilized_channels(template):
    """Returns the channels that are utilized in the template (List)"""
    for temp in template:
        non_zero_indices = [i for i, num in enumerate(temp) if num != 0]
        if non_zero_indices:
            return non_zero_indices


def extract_template(template_used_data_path, templateId):
    """Extracts the template from the template_used_data_path 
    (Returned object be something like channel counts (383) X timestamps (82))"""
    template = np.load(template_used_data_path)
    return template[templateId]

# def compare_spike_count_similarity(computed_spike, kilosort_spike):
#     # Need to convert computed spike into -3 and 3 timescale
#     """ OUTDATED) - this is just looking at cca,
#     This needs to be updated to use the new spike count per bin"""
#     computed_spike = np.array(computed_spike)
#     kilosort_spike = np.array(kilosort_spike)

#     c = np.correlate(computed_spike, kilosort_spike)
#     cross_corr = c / np.sqrt(np.sum(computed_spike ** 2)
#                              * np.sum(kilosort_spike ** 2))
#     return cross_corr


def plot_recovered_spike_against_kilosort_single_trial(computed_spike, kilosort_spike, neuron_id, trial_num, threshold, save_folder_path=None, multivariate=False):
    """Plots the recovered spike against the kilosort spike, for a single trial"""
    if len(computed_spike) == 0:
        if multivariate:
            print("computed_spike is empty for multivariate")
        else:
            print("computed_spike is empty for univariate ")
        return
    cross_corr = druckmann_correlation_calculation(
        computed_spike, kilosort_spike)
    fig, ax1 = plt.subplots()
    for i, spike in enumerate(computed_spike):
        if i == 0:
            if multivariate:
                ax1.vlines(spike, 0, 1, color='blue',
                           label='Multi-variate recovered Spike')
            else:
                ax1.vlines(spike, 0, 1, color='blue',
                           label='Uni-variate recovered Spike')
        else:
            ax1.vlines(spike, 0, 1, color='blue')
    for i, spike in enumerate(kilosort_spike):
        if i == 0:
            ax1.vlines(spike, 0, 1, color='red',
                       label="kilosort computed spikes")
        else:
            ax1.vlines(spike, 0, 1, color='red')
    if multivariate:
        title = f"multivariate_Neuron_{neuron_id}_Trial_{trial_num}_Threshold {threshold}_ccr_{round(np.mean(cross_corr),2)}_spike_count_comparison"
    else:
        title = f"univariate_Neuron_{neuron_id}_Trial_{trial_num}_Threshold {threshold}_ccr_{round(np.mean(cross_corr),2)}spike_count_comparison"
    plt.title(title)
    plt.legend()
    if save_folder_path:
        plt.savefig(os.path.join(save_folder_path, title + ".png"))
    plt.show()


def convert_spike_time_to_fr(spikeTimes, config, rate=True):
    '''Input: list of spikeTimes
    Output: firing rates using the defined bin_width and stride (which might be different from the original bim of 86 timestamps)'''
    spikeTimes = np.array(spikeTimes)
    # Use Yi's code to identify a firing rate based on a new window_size and stride
    TIME_BEFORE_CUE = config['TIME_BEFORE_CUE']
    TIME_AFTER_CUE = config['TIME_AFTER_CUE']
    bin_width = config['bin_width']
    stride = config['stride']

    bin_centers, bin_fr = sliding_histogram(
        spikeTimes, TIME_BEFORE_CUE, TIME_AFTER_CUE, bin_width, stride, rate=True)

    return bin_centers, bin_fr


def convert_spike_time_to_fr_multiple_trials(spikeTimes_multiple_trial, config, rate=True):
    '''Input: list of trials of spiketimes, shape (num_trials, num_spiketimes)
    Output: list of trials of fr, shape (num_trials, num_bins)
    firing rates using the defined bin_width and stride (which might be different from the original bim of 86 timestamps)'''
    final_bin_centers = []
    final_bin_fr = []
    for spike_time in spikeTimes_multiple_trial:
        bin_centers, bin_fr = convert_spike_time_to_fr(
            spike_time, config, rate=True)
        final_bin_centers.append(bin_centers)
        final_bin_fr.append(bin_fr)
    return final_bin_centers, final_bin_fr


def pearson_fr_calculation(computed_fr, kilosort_fr):
    return scipy.stats.pearsonr(computed_fr, kilosort_fr)[0]


def elephant_correlation_calculation(computed_spike, kilosort_spike):
    '''Takes an input of 2 spike trains and returns the correlation coefficient'''
    spike_train_1 = neo.SpikeTrain(
        computed_spike, units='s', t_start=-3, t_stop=3)
    spike_train_2 = neo.SpikeTrain(
        kilosort_spike, units='s', t_start=-3, t_stop=3)
    correlation_coeff = spike_time_tiling_coefficient(
        spike_train_1, spike_train_2)
    return correlation_coeff


def plot_fr_comparison(univariate_bin_fr, multivariate_bin_fr, kilosort_bin_fr, NEURON_ID, threshold, trialNum, num_of_averaged_trials, desired_trial_type_name=None, save_folder_path=None):
    '''univariate_bin_fr, multivariate_bin_fr, and kilosort_bin_fr should have the same size (since they are both binned up for 6 seconds)
    Input should be the firing rate for one trial, and should have the shape of (num_bins, 1)'''
    x = np.arange(len(univariate_bin_fr))
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))

    # Plot for univariate_bin_fr
    if trialNum is None and num_of_averaged_trials and desired_trial_type_name:
        title = f"Neuron_{NEURON_ID}_Threshold {threshold}_fr_comparison_averaged_across_{num_of_averaged_trials}_trials_for_trial_type_{desired_trial_type_name}"
    else:
        title = f"Neuron_{NEURON_ID}_Trial_{trialNum}_Threshold {threshold}_fr_comparison"
    axs[0].set_title('univariate retrieved spikes' + title)
    axs[0].plot(x, univariate_bin_fr, color="red")
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Firing rate')

    # Plot for multivariate_bin_fr
    axs[1].set_title('multivariate retrieved spikes' + title)
    axs[1].plot(x, multivariate_bin_fr, color="green")
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Firing rate')

    # Plot for kilosort_bin_fr
    axs[2].set_title('kilosort retrieved spikes' + title)
    axs[2].plot(x, kilosort_bin_fr, color="blue")
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Firing rate')

    plt.tight_layout()
    if save_folder_path:
        plt.savefig(os.path.join(save_folder_path, title + ".png"))

    plt.show()

# TODO: 1. only look at non-stimulation trials, 2. look at the difference between the raw_voltage for the drifted and non-drifted trials. Did the SNR decrease? Or did the majority of the signals decreased? And how did the projections of the univariate projections change? 3. Implmeent the multi-NOVA multivariate method


# def plot_uni_multi_kilo_fr_across_specified_trials_specified_neuron(list_of_trial_Id, desired_trial_type_name, raw_spike_data_path, neuron_identity_data_path, trial_time_info_path, NEURON_ID, TIME_BEFORE_CUE, TIME_AFTER_CUE, template_id, bin_width, threshold, stride, peakChannel):
#     '''Given a list of trials, and a specific neuron ID, calcualte the uni, multi, kilosort firing rate against one another, and then plot them'''

#     kilosort_bin_fr_across_trials = []
#     univariate_bin_fr_across_trials = []
#     multivariate_bin_fr_across_trials = []

#     for trialNum in list_of_trial_Id:
#         # Grab original kilosort retrieved spikes for this trial
#         kilosort_computed_spike_time, cue_time = extract_single_neuron_spike_times_for_specific_trial(
#             raw_spike_data_path, neuron_identity_data_path, trial_time_info_path, trialNum, NEURON_ID, TIME_BEFORE_CUE, TIME_AFTER_CUE)
#         kilosort_computed_spike_time = center_around_cue(
#             kilosort_computed_spike_time, cue_time)

#         # Grab the template and raw_voltage data used for this neuron, for the specified neuron ID
#         current_neuron_template = extract_template(
#             template_used_data_path, template_id)
#         raw_voltage_data_per_neuron = getDataAndPlot(
#             voltage_data_path, cue_time + TIME_BEFORE_CUE, cue_time + TIME_AFTER_CUE, peakChannel, plot=False, allChannels=True)
#         utilized_Channels = return_utilized_channels(current_neuron_template)

#         # Convert the raw_voltage data into a univariate projection
#         univariate_projection = univariate(
#             raw_voltage_data_per_neuron, current_neuron_template)

#         # Convert the raw_voltage data into a multivariate projection
#         multivariate_projection = multivariate(
#             raw_voltage_data_per_neuron, current_neuron_template, utilized_Channels)

#         # Convert the projection into spikes
#         univariate_recovered_spike = convert_dot_product_to_spike_count_univariate(
#             univariate_projection, utilized_Channels, threshold)
#         multivariate_recovered_spike = convert_dot_product_to_spike_count_multivariate(
#             multivariate_projection, utilized_Channels, threshold)

#         # Convert the spike count into firing rate
#         bin_centers, univariate_bin_fr = convert_spike_time_to_fr(
#             univariate_recovered_spike, TIME_BEFORE_CUE, TIME_AFTER_CUE, bin_width, stride, rate=True)
#         _, multivariate_bin_fr = convert_spike_time_to_fr(
#             multivariate_recovered_spike, TIME_BEFORE_CUE, TIME_AFTER_CUE, bin_width, stride, rate=True)
#         _, kilosort_bin_fr = convert_spike_time_to_fr(
#             kilosort_computed_spike_time, TIME_BEFORE_CUE, TIME_AFTER_CUE, bin_width, stride, rate=True)

#         kilosort_bin_fr_across_trials.append(kilosort_bin_fr)
#         univariate_bin_fr_across_trials.append(univariate_bin_fr)
#         multivariate_bin_fr_across_trials.append(multivariate_bin_fr)
#         print(f'{trialNum} trial done')
#     print(f'All trials done')
#     kilosort_bin_fr_across_trials = np.array(kilosort_bin_fr_across_trials)
#     univariate_bin_fr_across_trials = np.array(univariate_bin_fr_across_trials)
#     multivariate_bin_fr_across_trials = np.array(
#         multivariate_bin_fr_across_trials)

#     print(
#         f'kilosort_bin_fr_across_trials shape: {kilosort_bin_fr_across_trials.shape}')
#     print(
#         f'univariate_bin_fr_across_trials shape: {univariate_bin_fr_across_trials.shape}')
#     print(
#         f'multivariate_bin_fr_across_trials shape: {multivariate_bin_fr_across_trials.shape}')

#     kilosort_bin_fr_across_trials = np.mean(
#         kilosort_bin_fr_across_trials, axis=0)
#     univariate_bin_fr_across_trials = np.mean(
#         univariate_bin_fr_across_trials, axis=0)
#     multivariate_bin_fr_across_trials = np.mean(
#         multivariate_bin_fr_across_trials, axis=0)

#     print(
#         f'kilosort_bin_fr_across_trials shape: {kilosort_bin_fr_across_trials.shape}')
#     print(
#         f'univariate_bin_fr_across_trials shape: {univariate_bin_fr_across_trials.shape}')
#     print(
#         f'multivariate_bin_fr_across_trials shape: {multivariate_bin_fr_across_trials.shape}')

#     plot_fr_comparison(univariate_bin_fr_across_trials, univariate_bin_fr_across_trials, kilosort_bin_fr_across_trials, NEURON_ID,
#                        threshold, num_of_averaged_trials=len(list_of_trial_Id), desired_trial_type_name=desired_trial_type_name, trialNum=None)

#     return univariate_bin_fr_across_trials, multivariate_bin_fr_across_trials, kilosort_bin_fr_across_trials


def plot_drift_special(NEURON_ID, peakChannel, template_id, threshold):

    drifted_trial_num = 302  # Licks left
    # Grab original kilosort retrieved spikes
    kilosort_computed_spike_time, cue_time = extract_single_neuron_spike_times_for_specific_trial(
        raw_spike_data_path, neuron_identity_data_path, trial_time_info_path, drifted_trial_num, NEURON_ID, TIME_BEFORE_CUE, TIME_AFTER_CUE)
    kilosort_computed_spike_time = center_around_cue(
        kilosort_computed_spike_time, cue_time)

    # Grab the template and raw_voltage data used for this neuron, for the specified neuron ID
    current_neuron_template = extract_template(
        template_used_data_path, template_id)
    raw_voltage_data_per_neuron = getDataAndPlot(
        voltage_data_path, cue_time + TIME_BEFORE_CUE, cue_time + TIME_AFTER_CUE, peakChannel, plot=False, allChannels=True)
    utilized_Channels = return_utilized_channels(current_neuron_template)

    # Convert the raw_voltage data into a univariate projection
    univariate_projection = univariate(
        raw_voltage_data_per_neuron, current_neuron_template)
    print("univariate_projection", univariate_projection)

    # Convert the raw_voltage data into a multivariate projection
    multivariate_projection = multivariate(
        raw_voltage_data_per_neuron, current_neuron_template, utilized_Channels)
    print("multivariate_projection", multivariate_projection)

    # Convert the projection into spikes
    univariate_recovered_spike = convert_dot_product_to_spike_count_univariate(
        univariate_projection, utilized_Channels, threshold)
    multivariate_recovered_spike = convert_dot_product_to_spike_count_multivariate(
        multivariate_projection, utilized_Channels, threshold)

    print("kilosort_computed_spike_time", len(
        kilosort_computed_spike_time), kilosort_computed_spike_time)
    print("univariate_recovered_spike", len(
        univariate_recovered_spike), univariate_recovered_spike)
    print("multivariate_recovered_spike", len(
        multivariate_recovered_spike), multivariate_recovered_spike)

    no_drift_trial_num = 150  # Licks left
    no_drift_kilosort_computed_spike_time, no_drift_cue_time = extract_single_neuron_spike_times_for_specific_trial(
        raw_spike_data_path, neuron_identity_data_path, trial_time_info_path, no_drift_trial_num, NEURON_ID, TIME_BEFORE_CUE, TIME_AFTER_CUE)
    no_drift_kilosort_computed_spike_time = center_around_cue(
        no_drift_kilosort_computed_spike_time, no_drift_cue_time)

    # Plot the spikes individually then against the kilosort recovered spikes
    # plot_spike_times_per_trial(univariate_recovered_spike, NEURON_ID, trialNum, threshold)
    # plot_spike_times_per_trial(multivariate_recovered_spike, NEURON_ID, trialNum, threshold, multivariate=True)

    elephant_corr = elephant_correlation_calculation(
        univariate_recovered_spike, kilosort_computed_spike_time)
    print(f'univariate elephant_corr: {elephant_corr}')
    elephant_corr = elephant_correlation_calculation(
        multivariate_recovered_spike, kilosort_computed_spike_time)
    print(f'multivariate elephant_corr: {elephant_corr}')
    # plot_recovered_spike_against_kilosort_single_trial(univariate_recovered_spike, kilosort_computed_spike_time, NEURON_ID, trialNum, threshold, save_folder_path)
    # plot_recovered_spike_against_kilosort_single_trial(multivariate_recovered_spike, kilosort_computed_spike_time, NEURON_ID, trialNum, threshold, save_folder_path, multivariate=True)

    fig, ax1 = plt.subplots()
    for i, spike in enumerate(univariate_recovered_spike):
        if i == 0:
            ax1.vlines(spike, 0, 0.25, color='blue',
                       label='Uni-variate recovered Spike')
        else:
            ax1.vlines(spike, 0, 0.25, color='blue')
    for i, spike in enumerate(multivariate_recovered_spike):
        if i == 0:
            ax1.vlines(spike, 0.25, 0.5, color='green',
                       label='Multi-variate recovered Spike')
        else:
            ax1.vlines(spike, 0.25, 0.5, color='green')

    for i, spike in enumerate(kilosort_computed_spike_time):
        if i == 0:
            ax1.vlines(spike, 0.5, 0.75, color='red',
                       label="kilosort computed spikes")
        else:
            ax1.vlines(spike, 0.5, 0.75, color='red')
    for i, spike in enumerate(no_drift_kilosort_computed_spike_time):
        if i == 0:
            ax1.vlines(spike, 0.75, 1, color='purple',
                       label="NO-DRIFT-Trial-150-kilosort computed spikes")
        else:
            ax1.vlines(spike, 0.75, 1, color='purple')

    title = f"NEURON_ID: {NEURON_ID} spike_count_comparison_for_drift (302) v.s. no drift (150)"
    plt.title(title)
    plt.legend()
    plt.show()


def generate_desired_trials(config):
    '''Generate the desired trial index based on the desired optogenetics and lick direction type'''
    # open the optogenetic stimualtion trial type file
    matFile = loadmat(config['trial_type_path'])
    optogenetic_trial_type = matFile['SessionData'][0, 0]["StimTypes"][0]
    optogenetic_desired_trial_type = config['optogenetic_desired_trial_type']
    optogenetic_desired_trial_idx = np.where(
        optogenetic_trial_type == optogenetic_desired_trial_type)[0]

    print("optogenetic_trial_type", optogenetic_trial_type)
    print("total length of optogenetic_trial_type", len(optogenetic_trial_type))
    print("number of trials with no stimulation",
          np.where(optogenetic_trial_type == 0)[0].shape)

    # open the lick direction trial type file
    lick_direction_trial_type = matFile['SessionData'][0, 0]["TrialTypes"][0]
    # TODO: Not sure if 0 is left lick or right lick, but assuming left lick
    lick_direction_desired_trial_type = config['lick_direction_desired_trial_type']
    lick_direction_desired_trial_idx = np.where(
        lick_direction_trial_type == lick_direction_desired_trial_type)[0]

    desired_trial_idx = np.intersect1d(
        optogenetic_desired_trial_idx, lick_direction_desired_trial_idx)

    return desired_trial_idx


def plot_spikes_for_selected_trials(config):
    '''Plots the spikes '''
    data = config['selected_trials_spikes_fr_voltage']

    fig, axs = plt.subplots(3, 1)
    fig.tight_layout(pad=1.0)  # Increase padding between subplots
    fig.suptitle(
        f"NEURON_ID: {config['NEURON_ID']} {config['desired_trial_type_name']}")

    for i, trial_id in enumerate(data.keys(), start=1):
        univariate_spike_time = data[trial_id]['univariate_spike_time']
        y = [i] * len(univariate_spike_time)
        axs[0].plot(univariate_spike_time, y, '|', color='blue')
        axs[0].set_title('Univariate')
    axs[0].set_yticks(range(1, len(data) + 1))
    axs[0].set_yticklabels(list(data.keys()))

    for i, trial_id in enumerate(data.keys(), start=1):
        multivariate_spike_time = data[trial_id]['multivariate_spike_time']
        y = [i] * len(multivariate_spike_time)
        axs[1].plot(multivariate_spike_time, y, '|', color='green')
        axs[1].set_title('Multivariate')
    axs[1].set_yticks(range(1, len(data) + 1))
    axs[1].set_yticklabels(list(data.keys()))

    for i, trial_id in enumerate(data.keys(), start=1):
        kilosort_spike_time = data[trial_id]['kilosort_spike_time']
        y = [i] * len(kilosort_spike_time)
        axs[2].plot(kilosort_spike_time, y, '|', color='black')
        axs[2].set_title('Kilosort')
    axs[2].set_yticks(range(1, len(data) + 1))
    axs[2].set_yticklabels(list(data.keys()))

    fig.text(0.04, 0.5, 'Trial ID', va='center', rotation='vertical')
    plt.show()


def plot_averaged_voltage_for_selected_trials(config):
    selected_trials_spikes_fr_voltage = config['selected_trials_spikes_fr_voltage']

    # Identifying the utilized_channels
    cur_template = extract_template(
        template_used_data_path=config["template_used_data_path"], templateId=config["NEURON_ID"])
    utilized_channels = np.array(return_utilized_channels(cur_template))

    fig, axs = plt.subplots(len(selected_trials_spikes_fr_voltage), 1, figsize=(
        10, 0.5*len(selected_trials_spikes_fr_voltage)))

    for i, trial_idx in enumerate(selected_trials_spikes_fr_voltage):
        raw_voltage = np.array(
            selected_trials_spikes_fr_voltage[trial_idx]['raw_voltage'])

        selected_channels_voltage = raw_voltage[utilized_channels]
        averaged_channels_voltage = np.mean(
            selected_channels_voltage, axis=0).squeeze()

        time = np.linspace(config["TIME_BEFORE_CUE"], config["TIME_AFTER_CUE"], len(
            averaged_channels_voltage))

        axs[i].plot(time, averaged_channels_voltage,
                    label=f'{trial_idx}')
        # axs[i].set_title(f'Voltage vs Time for Trial {trial_idx}')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Voltage')
        # axs[i].set_yticks(np.arange(np.min(averaged_channels_voltage), np.max(averaged_channels_voltage), step=0.5))
        axs[i].legend()
    fig.suptitle(
        f"Averaged Voltage across Utilized Channel for neuron{config['NEURON_ID']}")
    plt.tight_layout()
    plt.show()


def get_voltage_stats(config):
    selected_trials_spikes_fr_voltage = config['selected_trials_spikes_fr_voltage']
    # Identifying the utilized_channels
    cur_template = extract_template(
        template_used_data_path=config["template_used_data_path"], templateId=config["NEURON_ID"])
    utilized_channels = np.array(return_utilized_channels(cur_template))

    result = {}
    for idx, trial_idx in enumerate(selected_trials_spikes_fr_voltage):
        raw_voltage = np.array(
            selected_trials_spikes_fr_voltage[trial_idx]['raw_voltage'])

        selected_channels_voltage = raw_voltage[utilized_channels]
        averaged_channels_voltage = np.mean(
            selected_channels_voltage, axis=0).squeeze()
        average = np.mean(averaged_channels_voltage)
        std = np.std(averaged_channels_voltage)

        result[trial_idx] = {
            "average": average,
            "std": std,
            "max": np.max(averaged_channels_voltage),
            "min": np.min(averaged_channels_voltage)
        }
        num_outliers = len([i for i in averaged_channels_voltage if (
            i > average + 3 * std) or (i < average - 3 * std)])
        result[trial_idx]["num_outliers"] = num_outliers
    return result


def plot_voltage_stats(config):
    # List of channels
    voltage_stats = config['voltage_stats']
    trials = list(voltage_stats.keys())
    positions = list(range(len(trials)))

    # Lists of metrics
    averages = [voltage_stats[trial]['average'] for trial in trials]
    std_devs = [voltage_stats[trial]['std'] for trial in trials]
    max_vals = [voltage_stats[trial]['max'] for trial in trials]
    min_vals = [voltage_stats[trial]['min'] for trial in trials]
    num_outliers = [voltage_stats[trial]['num_outliers'] for trial in trials]

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(5, figsize=(5, 10))

    # Plot the data
    axs[0].bar(positions, averages, color='blue')
    axs[0].set_title('Average')
    axs[0].set_ylabel('Value')
    axs[0].set_xticks(positions)
    axs[0].set_xticklabels(trials, rotation='vertical')

    axs[1].bar(positions, std_devs, color='orange')
    axs[1].set_title('Standard Deviation')
    axs[1].set_ylabel('Value')
    axs[1].set_xticks(positions)
    axs[1].set_xticklabels(trials, rotation='vertical')

    axs[2].bar(positions, max_vals, color='green')
    axs[2].set_title('Maximum Values')
    axs[2].set_ylabel('Value')
    axs[2].set_xticks(positions)
    axs[2].set_xticklabels(trials, rotation='vertical')

    axs[3].bar(positions, min_vals, color='red')
    axs[3].set_title('Minimum Values')
    axs[3].set_ylabel('Value')
    axs[3].set_xticks(positions)
    axs[3].set_xticklabels(trials, rotation='vertical')

    axs[4].bar(positions, num_outliers, color='purple')
    axs[4].set_title('Outliers')
    axs[4].set_ylabel('Value')
    axs[4].set_xticks(positions)
    axs[4].set_xticklabels(trials, rotation='vertical')

    # Display the plot
    fig.suptitle(
        f'Voltage Stats for utilized Channels of neuron{config["NEURON_ID"]} {config["desired_trial_type_name"]}')
    # plt.subplots_adjust(bottom=0.1, top=0.4)
    plt.show()


def heatmap_univariate(config):
    n = len(config["selected_trials"])  # number of subfigures
    fig, axs = plt.subplots(1, n, figsize=(n*3, 3))

    for ax, trial_idx in zip(axs, config["selected_trials"]):
        # shape should be (num_time_bins for 6 seconds, num_channels)
        univariate_projection = config["selected_trials_spikes_fr_voltage"][trial_idx]["univariate_projection"]

        # Identifying the utilized_channels
        cur_template = extract_template(
            template_used_data_path=config["template_used_data_path"], templateId=config["NEURON_ID"])
        utilized_channels = np.array(return_utilized_channels(cur_template))
        univariate_projection = univariate_projection[:, utilized_channels].transpose(
        )

        img = ax.imshow(univariate_projection,
                        interpolation='nearest', aspect='auto', cmap='seismic')
        fig.colorbar(img, ax=ax)
        ax.set_title(f'{trial_idx}')

    plt.tight_layout()
    plt.show()


def main():
    # time_for_all_trials = 4193 #seconds took to record all trials
    # sampling rate is 30000 Hz
    config = {
        'NEURON_ID': 274,
        'threshold': 0.1,
        'bin_width': 0.01,
        'stride': 0.01,
        'save_folder_path': "./neuron_365/",
        'selected_trials': [152, 169],
        'raw_spike_data_path': 'imec1_ks2/spike_times_sec_adj.npy',
        'neuron_identity_data_path': 'imec1_ks2/spike_clusters.npy',
        'raw_voltage_data_path': "./NL_NL106_20221103_session1_g0_tcat.imec1.ap.bin",
        'neuron_channel_path': "imec1_ks2/waveform_metrics.csv",
        'trial_time_info_path': 'AccessarySignalTime.mat',
        'template_used_data_path': 'imec1_ks2/templates.npy',
        'trial_type_path': './NL106_yes_no_multipole_delay_stimDelay_Nov03_2022_Session1.mat',
        't0': 460,
        't1': 470,
        'TIME_BEFORE_CUE': -3,
        'TIME_AFTER_CUE': 3,
        'trialNum': 152,
        'optogenetic_desired_trial_type': 0,  # no optogenetic stimulation
        'lick_direction_desired_trial_type': 1,  # 0 left lick, 1 right lick
    }
    config['desired_trial_type_name'] = f'optogenetic stimulation type (f{config["optogenetic_desired_trial_type"]}) + lick direction ({config["lick_direction_desired_trial_type"]}) threshold {config["threshold"]}'
    config['template_id'] = config['NEURON_ID']
    config['peakChannel'] = extractPeakChannel(config['NEURON_ID'])
    config['selected_trials'] = generate_desired_trials(config)[-6:]
    # additional_trials = list(range(200, 210))
    # config['selected_trials'] = np.sort(np.concatenate(
    #     (config['selected_trials'], additional_trials)))  # add 10 more trials that drifted
    print(config['selected_trials'])
    # list_of_threshold = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    # list_of_neuron_id = extract_processed_neuron_raster(raw_spike_data_path, neuron_identity_data_path).keys()
    # list_of_trial_num = [i for i in range(1, 300)]

    # univariate_bin_fr_across_trials, multivariate_bin_fr_across_trials, kilosort_bin_fr_across_trials = bin_fr_calculation_across_specified_trials_specified_neuron(list_of_trial_Id=[32, 33, 34, 35, 36], desired_trial_type_name = desired_trial_type_name, raw_spike_data_path=raw_spike_data_path, neuron_identity_data_path=neuron_identity_data_path, trial_time_info_path=trial_time_info_path
    #                                                            , NEURON_ID=NEURON_ID, TIME_BEFORE_CUE=TIME_BEFORE_CUE, TIME_AFTER_CUE=TIME_AFTER_CUE, template_id=template_id, bin_width=bin_width, threshold=threshold, stride=stride, peakChannel=peakChannel)

    config["selected_trials_spikes_fr_voltage"] = get_uni_multi_kilosort_spikes_across_selected_trials(
        config)
    # print("uni", config["selected_trials_spikes_fr_voltage"]
    #       [152]["univariate_projection"].shape)
    # print("multi", config["selected_trials_spikes_fr_voltage"]
    #       [152]["multivariate_projection"].shape)
    heatmap_univariate(config)

    # plot_spikes_for_selected_trials(config)
    # plot_averaged_voltage_for_selected_trials(config)
    # config["voltage_stats"] = get_voltage_stats(config)
    # print(config["voltage_stats"])
    # plot_voltage_stats(config)
    # print("result.keys()", result.keys())
    # print("result['univariate_spike_time'].shape",
    #       result[result.keys[0]]['univariate_spike_time'].shape)
    # print("result['multi_variate_spike_time'].shape",
    #       result[result.keys[0]]['multi_variate_spike_time'].shape)
    # print("result['kilosort_spike_time'].shape",
    #       result[result.keys[0]]['kilosort_spike_time'].shape)

    # # plot_drift_special(NEURON_ID, peakChannel, template_id, threshold)
    # # #Grab original kilosort retrieved spikes
    # kilosort_computed_spike_time, cue_time = extract_single_neuron_spike_times_for_specific_trial(
    #     config["raw_spike_data_path"], config["neuron_identity_data_path"], config["trial_time_info_path"], config["trialNum"], config["NEURON_ID"], config["TIME_BEFORE_CUE"], config["TIME_AFTER_CUE"])
    # kilosort_computed_spike_time = center_around_cue(
    #     kilosort_computed_spike_time, cue_time)

    # # #Grab the template and raw_voltage data used for this neuron, for the specified neuron ID
    # current_neuron_template = extract_template(
    #     config["template_used_data_path"], config['template_id'])
    # raw_voltage_data_per_neuron = getDataAndPlot(config["raw_voltage_data_path"], cue_time + config['TIME_BEFORE_CUE'],
    #                                              cue_time + config['TIME_AFTER_CUE'], config['peakChannel'], plot=False, allChannels=True)
    # utilized_Channels = return_utilized_channels(current_neuron_template)

    # # #Convert the raw_voltage data into a univariate projection
    # univariate_projection = univariate(
    #     raw_voltage_data_per_neuron, current_neuron_template)
    # print("univariate_projection", univariate_projection)

    # # Loop through univairatie projection and identify if there are any numbers larger than 0
    # # If there are, then that means that the neuron fired at that time

    # # Convert the raw_voltage data into a multivariate projection
    # print("input to multivariate function has shape, ",
    #       raw_voltage_data_per_neuron.shape)
    # multivariate_projection = multivariate(
    #     raw_voltage_data_per_neuron, current_neuron_template, utilized_Channels)
    # print("multivariate_projection", multivariate_projection)

    # #Convert the projection into spikes
    # univariate_recovered_spike = convert_dot_product_to_spike_count_univariate(univariate_projection, utilized_Channels, threshold)
    # multivariate_recovered_spike = convert_dot_product_to_spike_count_multivariate(multivariate_projection, utilized_Channels, threshold)

    # print("kilosort_computed_spike_time", len(kilosort_computed_spike_time), kilosort_computed_spike_time)
    # print("univariate_recovered_spike", len(univariate_recovered_spike), univariate_recovered_spike)
    # print("multivariate_recovered_spike", len(multivariate_recovered_spike), multivariate_recovered_spike)

    # # Plot the spikes individually then against the kilosort recovered spikes
    # plot_uni_multivariate_spike_count_per_trial(univariate_recovered_spike, NEURON_ID, trialNum, threshold)
    # plot_uni_multivariate_spike_count_per_trial(multivariate_recovered_spike, NEURON_ID, trialNum, threshold, multivariate=True)

    # plot_recovered_spike_against_kilosort_single_trial(univariate_recovered_spike, kilosort_computed_spike_time, NEURON_ID, trialNum, threshold, save_folder_path)
    # plot_recovered_spike_against_kilosort_single_trial(multivariate_recovered_spike, kilosort_computed_spike_time, NEURON_ID, trialNum, threshold, save_folder_path, multivariate=True)

    # #Since we're using a bin width of 0.01, and some of the neurons fire sporadically, the firing rate returned is in Hz, and it might be 100Hz, which is higher than the real firing rate

    # #Convert the spike count into firing rate
    # bin_centers, univariate_bin_fr = convert_spike_time_to_fr(univariate_recovered_spike, TIME_BEFORE_CUE, TIME_AFTER_CUE, bin_width, stride, rate=True)
    # _, multivariate_bin_fr = convert_spike_time_to_fr(multivariate_recovered_spike, TIME_BEFORE_CUE, TIME_AFTER_CUE, bin_width, stride, rate=True)
    # _, kilosort_bin_fr = convert_spike_time_to_fr(kilosort_computed_spike_time, TIME_BEFORE_CUE, TIME_AFTER_CUE, bin_width, stride, rate=True)

    # plot_fr_comparison(univariate_bin_fr, multivariate_bin_fr, kilosort_bin_fr, NEURON_ID,  threshold, trialNum, save_folder_path)

    # print("bin_centers", bin_centers)
    # print("bin_fr", univariate_bin_fr)
    # print("kilosort_bin_fr", kilosort_bin_fr)
    # print('Univariate Pearson correlation coefficient: ', scipy.stats.pearsonr(univariate_bin_fr, kilosort_bin_fr)[0])
    # print('Multivariate Pearson correlation coefficient: ', scipy.stats.pearsonr(multivariate_bin_fr, kilosort_bin_fr)[0])
    # print('univariate kilosort correlation druckmann equation:', druckmann_correlation_calculation(univariate_recovered_spike, kilosort_computed_spike_time))
    # print('multivariate kilosort correlation druckmann equation:', druckmann_correlation_calculation(multivariate_recovered_spike, kilosort_computed_spike_time))


# threshold_results, pearson_corr_threshold = find_optimal_threshold(list_of_threshold, list_of_neuron_id, list_of_trial_num, raw_spike_data_path, neuron_identity_data_path, trial_time_info_path, voltage_data_path, template_used_data_path, TIME_BEFORE_CUE, TIME_AFTER_CUE, bin_width, stride)

# print("threshold_results", threshold_results)
# print("pearson_corr_threshold", pearson_corr_threshold)
# np.save("./threshold_results.npy", threshold_results)
# np.save("./pearson_corr_threshold.npy", pearson_corr_threshold)
if __name__ == "__main__":
    main()


# def find_optimal_threshold(list_of_threshold, list_of_neuron_id, list_of_trial_num, raw_spike_data_path, neuron_identity_data_path, trial_time_info_path, voltage_data_path, template_used_data_path, TIME_BEFORE_CUE, TIME_AFTER_CUE, bin_width, stride):
#     results = {}
#     start_time = time.time()

#     pearson_corr_with_neuron = {}
#     for cur_threshold in list_of_threshold:

#         cur_threshold_pearson_corr = []
#         pearson_corr_with_neuron[cur_threshold] = {}

#         for cur_neuron_id in list_of_neuron_id:
#             pearson_corr_with_neuron[cur_threshold][cur_neuron_id] = []
#             cur_template_id = cur_neuron_id
#             peakChannel = extractPeakChannel(cur_neuron_id)

#             #grab the template used for this neuron
#             current_neuron_template = extract_template(template_used_data_path, cur_template_id)
#             utilized_Channels = return_utilized_channels(current_neuron_template)

#             for cur_trial_num in list_of_trial_num:

#                 #grav the raw spike data and center it around the go cue
#                 kilosort_computed_spike_time, cue_time = extract_single_neuron_spike_times_for_specific_trial(raw_spike_data_path, neuron_identity_data_path, trial_time_info_path, cur_trial_num, cur_neuron_id, TIME_BEFORE_CUE, TIME_AFTER_CUE)
#                 kilosort_computed_spike_time = center_around_cue(kilosort_computed_spike_time, cue_time)

#                 #grav raw_voltage_data_per_neuron_per specific trial
#                 raw_voltage_data_per_neuron = getDataAndPlot(voltage_data_path, cue_time + TIME_BEFORE_CUE, cue_time + TIME_AFTER_CUE, peakChannel, plot=False, allChannels=True)

#                 #grab the univariate data per neuron per trial
#                 univariate_projection = univariate(raw_voltage_data_per_neuron, current_neuron_template)

#                 #convert univariate projection into spike count per bin
#                 univariate_recovered_spike = convert_dot_product_to_spike_count_univariate(univariate_projection, utilized_Channels, cur_threshold)
#                 # cross_corr = compare_spike_count_similarity(univariate_recovered_spike, kilosort_computed_spike_time)
#                 bin_centers, univariate_bin_fr = convert_spike_time_to_fr(univariate_recovered_spike, TIME_BEFORE_CUE, TIME_AFTER_CUE, bin_width, stride, rate=True)
#                 #Since we're using a bin width of 0.01, the firing rate returned is in Hz, and it might be 100Hz, which is higher
#                 _, kilosort_bin_fr = convert_spike_time_to_fr(kilosort_computed_spike_time, TIME_BEFORE_CUE, TIME_AFTER_CUE, bin_width, stride, rate=True)
#                 pearson_corr = pearson_fr_calculation(univariate_bin_fr, kilosort_bin_fr)
#                 cur_threshold_pearson_corr.append(pearson_corr)

#             print(f"Elapsed time: {time.time() - start_time} seconds")
#             print(f"Threshold {cur_threshold} has a mean pearson correlation of {np.mean(cur_threshold_pearson_corr)} for neuron {cur_neuron_id}")
#             pearson_corr_with_neuron[cur_threshold][cur_neuron_id] = np.mean(cur_threshold_pearson_corr)

#         print(f"Elapsed time: {time.time() - start_time} seconds")
#         print(f"Threshold {cur_threshold} has a mean pearson correlation of {np.mean(cur_threshold_pearson_corr)}")
#         results[cur_threshold] = np.mean(cur_threshold_pearson_corr)
#     return results, pearson_corr_with_neuron

# def druckmann_correlation_calculation(computed_spike, kilosort_spike):
#     # Both input should be the times of the spikes between -3 and +3 seconds
#     precision_delta = 0.005  # 5ms

#     # Calculate the number of spikes that are within +- 2.5ms of each other (since we need to double 5ms)
#     coincidences_count = 0
#     # print("computed spike", len(computed_spike))
#     # print("kilosort spike", len(kilosort_spike))
#     # print("computed spike", computed_spike)
#     # print("kilosort spike", kilosort_spike)

#     for spike in computed_spike:
#         for real_spike in kilosort_spike:
#             if abs(spike - real_spike) <= precision_delta/2:
#                 coincidences_count += 1

#     # Calculate the expected number of coincidences generated by a homogeneous poisson process with the same rate f as the spike train kilosort_spike
#     real_fr = len(kilosort_spike)/6
#     # probability_of_zero_spike_in_window = math.e ** (-1 * real_fr * precision_delta)
#     # expected_coincidences = 1 - probability_of_zero_spike_in_window #This would include the case where there are 2 spikes, or 3 spikes, however, I would both count them as 1 coincidence
#     expected_coincidences = 2 * real_fr * precision_delta * len(computed_spike)

#     # print(f"coincidences_count: {coincidences_count}")
#     # print(f"expected_coincidences: {expected_coincidences}")

#     bottom = 0.5 * (len(computed_spike) + len(kilosort_spike))
#     top = (coincidences_count - expected_coincidences)
#     N = (1 - 2 * real_fr * precision_delta)
#     # print(f"top: {top}")
#     # print(f"bottom: {bottom}")
#     # print(f"N: {N}")
#     return top/(bottom * N)
