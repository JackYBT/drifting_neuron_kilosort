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
from extract_template import extract_template, return_utilized_channels, convert_spike_time_to_fr_multiple_trials
from spike_raster_display import extract_trial_time_info, extract_single_neuron_spike_time_across_trials, center_around_cue, sliding_histogram, extract_processed_neuron_raster


def generate_spike_time_from_bin(spike_count_per_bin):
    """Takes a list of spike counts per bin (as indicated by 0 as no spikes during that bin, and 1 is yes spike during that bin), 
    and convert to a list of spike times corresponding to the time of the bin, between -3 and +3 seconds"""
    time_arr_length = len(spike_count_per_bin)
    time_bin_length_in_sec = 6/time_arr_length
    result_arr = [-3 + i*time_bin_length_in_sec for i,
                  spike in enumerate(spike_count_per_bin) if spike == 1]

    return result_arr


def multivariate(raw_voltage, template, utilized_channels):
    """
    This function calculates normalized multi-variate analysis of raw voltage data using a template.
    """
    totalNumBins = raw_voltage.shape[1] / \
        template.shape[0]  # should be number of samples / 82

    # should have N (number of bins) x 1 (dot product of x and y)
    final_Multivariate = []

    for bin in range(int(totalNumBins)):
        # print("utilizied_channels", utilized_channels)
        # print("raw_voltage[utilized_channels, bin*template.shape[0]:(bin+1)*template.shape[0]]",
        #   raw_voltage[utilized_channels, bin*template.shape[0]:(bin+1)*template.shape[0]].shape)
        x = raw_voltage[utilized_channels, bin*template.shape[0]                        :(bin+1)*template.shape[0]].transpose().flatten()
        y = template[:, utilized_channels].flatten()
        # x and y must first be converted into 82 x 383 martix, select only the utilized Channels then flattened
        multivariate = np.dot(x, y)

        mag_x = np.linalg.norm(x)
        mag_y = np.linalg.norm(y)
        if mag_x != 0 and mag_y != 0:
            normalized_multivariate = multivariate/(mag_x*mag_y)
        else:
            normalized_multivariate = 0

        final_Multivariate.append(normalized_multivariate)

    return np.array(final_Multivariate)


def multiple_trial_multivariate(raw_voltage, template, utilized_channels):
    '''This would return the multivariate projections for multiple designated trials, 
    when given a list of raw_voltage to compute'''
    finalMultivariate = []
    for trial in range(raw_voltage.shape[0]):

        single_multi_projection = multivariate(
            raw_voltage[trial], template, utilized_channels)
        finalMultivariate.append(single_multi_projection)
    return np.array(finalMultivariate)


def univariate(raw_voltage, template):
    """
    This function calculates normalized univariate analysis of raw voltage data using a template.
    """
    totalNumBins = raw_voltage.shape[1] / \
        template.shape[0]  # should be number of samples / 82

    # should have N (number of bins) x 383 (number of channales)
    finalUnivariate = []

    # TODO: I should have only selected "utilized_channels" in the first place, but I did it later on in the code that it doesn't affect
    # The final output. But i should fix for clarity.
    for bin in range(int(totalNumBins)):
        current_bin_prediction = []
        # current_bin_voltage should have shape 383 X 82
        current_bin_voltage = raw_voltage[:, bin *
                                          template.shape[0]:(bin+1)*template.shape[0]]
        # print("current_bin_voltage", current_bin_voltage.shape)
        for channel in range(current_bin_voltage.shape[0]):
            x = current_bin_voltage[channel, :]
            y = template[:, channel]
            univariate = np.dot(x, y)

            mag_x = np.linalg.norm(x)
            mag_y = np.linalg.norm(y)
            if mag_x != 0 and mag_y != 0:
                normalized_univariate = univariate/(mag_x*mag_y)
            else:
                normalized_univariate = 0

            current_bin_prediction.append(normalized_univariate)

        finalUnivariate.append(current_bin_prediction)

    return np.array(finalUnivariate)


def multiple_trial_univariate(raw_voltage, template):
    '''This would return the univariate projections for multiple designated trials, 
    when given a list of raw_voltage to compute'''
    finalUnivariate = []
    for trial in range(raw_voltage.shape[0]):
        single_uni_projection = univariate(raw_voltage[trial], template)
        finalUnivariate.append(single_uni_projection)
    return np.array(finalUnivariate)


# def plot_univariate(univariate_data, utilized_channels):
#     """Plots the raw projection of the voltage data onto the template, for a single trial"""
#     for channel in utilized_channels:
#         plt.plot(univariate_data[:, channel], label='Channel ' + str(channel))

#     plt.ylabel('Normalized Univariate')
#     plt.xlabel('#bins (each bin is 82 timepoints)')
#     plt.legend()
#     plt.show()


def convert_dot_product_to_spike_count_univariate(univariate_data, utilized_Channels, threshold):
    """Converts the projections of the voltage data onto the template into spike counts, for a single 
    trial of univariate_data."""
    spike_count_per_bin = []
    for bin in range(univariate_data.shape[0]):
        # TODO: I originally did if np.mean(univariate_data[bin][utilized_Channels]) > threshold, and it performed much better than the current implementation
        if (np.mean(univariate_data[bin][utilized_Channels]) > threshold):
            spike_count_per_bin.append(1)
        else:
            spike_count_per_bin.append(0)
        # if all(i > threshold for i in univariate_data[bin][utilized_Channels]):
        #     spike_count_per_bin.append(1)
        #     # print("bin", bin, "mean", np.mean(univariate_data[bin][utilized_Channels]))
        # else:
        #     spike_count_per_bin.append(0)
    # returns a list of spike counts per bin, each count is separated out by 82 timepoints
    # print("spike_count_per_bin", len(spike_count_per_bin))
    # print("spike_count_per_bin", spike_count_per_bin)

    spike_time = generate_spike_time_from_bin(spike_count_per_bin)
    return spike_time


def convert_dot_product_to_spike_count_multivariate(multivariate_projection, utilized_Channels, threshold):
    '''converts the projections of multi-varaite data onto the template into spike counts, for a single trial of 
    multi-variate data. Returns a list of spike times.'''
    spike_count_per_bin = []
    for bin in range(multivariate_projection.shape[0]):
        if multivariate_projection[bin] > threshold:
            spike_count_per_bin.append(1)
        else:
            spike_count_per_bin.append(0)
    spike_time = generate_spike_time_from_bin(spike_count_per_bin)
    return spike_time


def convert_dot_product_to_spike_count_univariate_multiple_trials(univariate_data, utilized_Channels, threshold):
    '''converts projections of uni-variate data into spike times for multiple trials. Returns an array of shape (num_Trials, num_spikes), where each row is a list of spike times for that trial.'''
    finalSpikeCount = []
    for trial in range(univariate_data.shape[0]):
        spike_count_per_trial = convert_dot_product_to_spike_count_univariate(
            univariate_data[trial], utilized_Channels, threshold)
        print("spike_time_per_trial", len(spike_count_per_trial))
        finalSpikeCount.append(spike_count_per_trial)
    return finalSpikeCount


def convert_dot_product_to_spike_count_multivariate_multiple_trials(multivariate_data, utilized_Channels, threshold):
    '''converts projections of multi-variate data into spike times for multiple trials. Returns an array of shape (num_Trials, num_spikes), where each row is a list of spike times for that trial.'''
    finalSpikeCount = []
    for trial in range(multivariate_data.shape[0]):
        spike_count_per_trial = convert_dot_product_to_spike_count_multivariate(
            multivariate_data[trial], utilized_Channels, threshold)
        finalSpikeCount.append(spike_count_per_trial)
    return finalSpikeCount


def plot_spike_times_per_trial(spike_times, NEURON_ID, trialNum, threshold, multivariate=False):
    """Takes a list of spike times in the trial, and plot the spikes for an individual trial"""
    if len(spike_times) == 0:
        if multivariate:
            print("spike_count_per_bin is empty for multivariate")
        else:
            print("spike_count_per_bin is empty for univariate ")
        return
    # print("spike_count_per_bin", len(spike_count_per_bin))
    for spike_time in spike_times:
        plt.axvline(x=spike_time, color='b')

    plt.axvline(x=0, color='r', label='Go Cue')
    plt.legend()
    plt.ylabel('Spike Count')
    plt.xlabel('seconds')
    if multivariate:
        title = f'Multivariate Recovered NEURON: {NEURON_ID} Trial ' + str(
            trialNum) + f' threshold:{threshold}'
    else:
        title = f'Univariate Recovered NEURON: {NEURON_ID} Trial ' + str(
            trialNum) + f' threshold:{threshold}'
    plt.title(title)
    plt.show()


def get_uni_multi_kilosort_spikes_across_selected_trials(config):
    '''Returns a dictionary with each idx being the actual trial ID, 
    and then the dictionary keys being a list of spike times or fire rates for that trial (uni multi and kilosort).
    Returns a total of 6 values per trial: uni, multi, kilo spiketimes, and their respective firing rates.'''
    NEURON_ID = config['NEURON_ID']
    raw_spike_data_path = config['raw_spike_data_path']
    neuron_identity_data_path = config['neuron_identity_data_path']
    trial_time_info_path = config['trial_time_info_path']
    threshold = config['threshold']
    listOfDesiredTrials = config['selected_trials']
    peakChannel = config['peakChannel']
    raw_voltage_data_path = config['raw_voltage_data_path']
    bin_width = config['bin_width']
    template_used_data_path = config['template_used_data_path']

    print("breakpoint 0")
    # 1. Extract the spike_times for kilosort for desired_trials
    all_spike_times = extract_processed_neuron_raster(
        raw_spike_data_path=raw_spike_data_path, neuron_identity_data_path=neuron_identity_data_path)
    _, all_cue_times = extract_trial_time_info(trial_time_info_path)
    temp_kilosort_spike_time = extract_single_neuron_spike_time_across_trials(
        all_spike_times, all_cue_times, NEURON_ID)

    # Only selecting the kilosort spike times for the desired trials
    final_kilosort_spike_time = [temp_kilosort_spike_time[i]
                                 for i in listOfDesiredTrials]
    print("breakpoint 1")
    # 2. Extracting the raw_voltage and templates for the desired trials
    raw_voltage_for_desired_trials = getMultipleTrialVoltage(
        listOfDesiredTrials, all_cue_times, raw_voltage_data_path, peakChannel)

    # Identify the shape of the template used for the desired neuron
    cur_template = extract_template(
        template_used_data_path=template_used_data_path, templateId=NEURON_ID)
    utilized_channels = return_utilized_channels(cur_template)

    print("breakpoint 2")
    # 3. Use the raw_voltage and templates to calculate univariate and multivariate spike times
    univariate_projection = multiple_trial_univariate(
        raw_voltage_for_desired_trials, cur_template)
    multivariate_projection = multiple_trial_multivariate(
        raw_voltage_for_desired_trials, cur_template, utilized_channels)

    univariate_spike_time = convert_dot_product_to_spike_count_univariate_multiple_trials(
        univariate_projection, utilized_channels, threshold)
    multi_variate_spike_time = convert_dot_product_to_spike_count_multivariate_multiple_trials(
        multivariate_projection, utilized_channels, threshold)

    # plot_uni_multi_kilosort_spikes_across_all_trials(
    #     listOfDesiredTrials[0], univariate_spike_time, multi_variate_spike_time, final_kilosort_spike_time, NEURON_ID, threshold)

    print("breakpoint 3")
    # 4. Convert the spike times to firing rate calculations
    _, univariate_fr = convert_spike_time_to_fr_multiple_trials(
        univariate_spike_time, config)
    _, multi_variate_fr = convert_spike_time_to_fr_multiple_trials(
        multi_variate_spike_time, config)
    _, kilosort_fr = convert_spike_time_to_fr_multiple_trials(
        final_kilosort_spike_time, config)

    result = {}
    print("braekpoint 4")
    for idx, trialNum in enumerate(listOfDesiredTrials):
        result[trialNum] = {
            "univariate_spike_time": univariate_spike_time[idx],
            "univariate_projection": univariate_projection[idx],
            "multivariate_spike_time": multi_variate_spike_time[idx],
            "multivariate_projection": multivariate_projection[idx],
            "kilosort_spike_time": final_kilosort_spike_time[idx],
            "univariate_fr": univariate_fr[idx],
            "multi_variate_fr": multi_variate_fr[idx],
            "kilosort_fr": kilosort_fr[idx],
            "raw_voltage": raw_voltage_for_desired_trials[idx]
        }
    return result


def plot_uni_multi_kilosort_spikes_across_all_trials(selected_trials, univariate_spike_time, multi_variate_spike_time, kilosort_spike_time, NEURON_ID, threshold):
    """Each input array needs to be a list of lists, where each list is the spike times for a single trial"""

    plt.figure(figsize=(10, 21))

    # First subplot
    plt.subplot(3, 1, 1)
    for i, trial in enumerate(univariate_spike_time):
        # Create a vertical line for each spike time in the trial
        plt.vlines(trial, i, i + 0.5, colors='red')
    # Set the y-axis limits
    plt.ylim(.5, len(univariate_spike_time) + .5)
    # plt.yticks(np.arange(0, len(univariate_spike_time) + selected_trials, step=1))
    # Set the x-axis label
    plt.xlabel('Spike time')
    # Set the y-axis label
    plt.ylabel('Trial')
    # Set the plot title
    plt.title(f'Univariate')

    plt.subplot(3, 1, 2)
    # For each individual trial
    for i, trial in enumerate(multi_variate_spike_time):

        # Create a vertical line for each spike time in the trial
        plt.vlines(trial, i, i + 0.5, colors='blue')
    # Set the y-axis limits
    plt.ylim(.5, len(multi_variate_spike_time) + .5)
    # Set the x-axis label
    plt.xlabel('Spike time')
    # Set the y-axis label
    plt.ylabel('Trial')
    # Set the plot title
    plt.title(f'Multivariate')

    plt.subplot(3, 1, 3)
    # For each individual trial
    for i, trial in enumerate(kilosort_spike_time):
        # Create a vertical line for each spike time in the trial
        plt.vlines(trial, i, i + 0.5, colors='k')
    # Set the y-axis limits
    plt.ylim(.5, len(kilosort_spike_time) + .5)
    # Set the x-axis label
    plt.xlabel('Spike time')
    # Set the y-axis label
    plt.ylabel('Trial')
    # Set the plot title
    plt.title(f'Kilosort Raster Plot')

    # Display the plot
    # for i, spike in enumerate(multi_variate_spike_time):
    #     if i == 0:
    #         ax1.vlines(spike, 0.25, 0.5, color='green', label='Multi-variate recovered Spike')
    #     else:
    #         ax1.vlines(spike, 0.25, 0.5, color='green')

    # for i, spike in enumerate(kilosort_spike_time):
    #     if i == 0:
    #         ax1.vlines(spike, 0.5, 0.75, color='red', label="kilosort computed spikes")
    #     else:
    #         ax1.vlines(spike, 0.5, 0.75, color='red')

    # title = f"NEURON_ID: {NEURON_ID} spike_count_comparison"
    # plt.title(title)
    # plt.legend()
    plt.suptitle(
        f'Raster Plot Comparison for {len(univariate_spike_time)} trials for Neuron {NEURON_ID} threshold: {threshold}')
    plt.tight_layout()
    plt.show()
