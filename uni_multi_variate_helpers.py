import numpy as np
from spike_extraction_helpers import get_neuron_spike_time_template_across_all_trials, center_around_cue, sliding_histogram, extract_processed_neuron_raster
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
from spike_extraction_helpers import extract_trial_time_info, center_around_cue, sliding_histogram, extract_processed_neuron_raster
import pywt


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


def convert_spike_time_to_time_bin_count(kilosort_spike_time, trialId, config):
    kilosort_spike_time = [(config['TIME_AFTER_CUE']+n) /
                           (2*config['TIME_AFTER_CUE'])for n in kilosort_spike_time]

    time_bins = np.zeros(
        len(config['univariate_projection_stats'][trialId]['denoised_projection']))

    for spike_time in kilosort_spike_time:
        bin_index = int(spike_time * len(time_bins))
        time_bins[bin_index] = 1
    return time_bins


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


def return_utilized_channels(template):
    """Returns the channels that are utilized in the template (List)"""
    for temp in template:
        non_zero_indices = [i for i, num in enumerate(temp) if num != 0]
        if non_zero_indices:
            return non_zero_indices


def extract_all_templates(config):
    template_used_data_path = config['template_used_data_path']
    template = np.load(template_used_data_path)
    return template  # Should contain all 1000 templates


def extract_template(config, templateId):
    """Extracts the template from the template_used_data_path
    (Returned object have shape (channel counts (383) X timestamps (82))"""
    return config['all_templates'][templateId]


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
        x = raw_voltage[utilized_channels, bin*template.shape[0]:(bin+1)*template.shape[0]].transpose().flatten()
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


def univariate(raw_voltage, template, final_scaling_amplitude_template):
    """
    This function calculates normalized univariate analysis of raw voltage data using a template.
    """
    totalNumBins = raw_voltage.shape[1] / \
        template.shape[0]  # should be number of samples / 82

    # should have N (number of bins) x 383 (number of channales)
    finalUnivariate = []

    # TODO: I should have only selected "utilized_channels" in the first place, but I did it later on in the code that it doesn't affect
    # The final output. But i should fix for clarity.

    # Since I need to scale the template without knowing the magnitude to scale it by, I can only scale it by the average of the entire trial
    # This is because the average of the entire trial should be the same as the average of the template, and therefore I can scale it by that
    template_scale_factor = np.mean(final_scaling_amplitude_template)
    print('template_scale_factor', template_scale_factor)
    template = template * template_scale_factor
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


def multiple_trial_univariate(raw_voltage, template, final_scaling_amplitude_template):
    '''This would return the univariate projections for multiple designated trials, 
    when given a list of raw_voltage to compute'''
    finalUnivariate = []
    for trial in range(raw_voltage.shape[0]):
        single_uni_projection = univariate(
            raw_voltage[trial], template, final_scaling_amplitude_template[trial])
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
    all_spikes = extract_processed_neuron_raster(config)
    _, all_cue_times = extract_trial_time_info(config)

    # shape(num_trial, num_spikes_in a specific trial) containing the adjusted times of the spikes
    temp_kilosort_spike_time, scaling_amplitude_template = get_neuron_spike_time_template_across_all_trials(
        all_spikes, all_cue_times, NEURON_ID)

    assert (len(temp_kilosort_spike_time) == len(all_cue_times)
            == len(scaling_amplitude_template) == 323)

    # Only selecting the kilosort spike times for the desired trials
    final_kilosort_spike_time = [temp_kilosort_spike_time[i]
                                 for i in listOfDesiredTrials]

    final_scaling_amplitude_template = [
        scaling_amplitude_template[i] for i in listOfDesiredTrials]

    # 2. Extracting the raw_voltage and templates for the desired trials
    raw_voltage_for_desired_trials = getMultipleTrialVoltage(
        listOfDesiredTrials, all_cue_times, raw_voltage_data_path, peakChannel)

    # # Identify the shape of the template used for the desired neuron
    cur_template = extract_template(
        config, templateId=NEURON_ID)
    utilized_channels = return_utilized_channels(cur_template)

    print("breakpoint 2")
    # 3. Use the raw_voltage and templates to calculate univariate and multivariate spike times
    univariate_projection = multiple_trial_univariate(
        raw_voltage_for_desired_trials, cur_template, final_scaling_amplitude_template)
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


def get_univariate_projection_stats(config):

    result = {}
    for trial_idx in config["selected_trials"]:
        result[trial_idx] = {}

        # Should be shape (num_time_bins for 6 seconds, num_channels)
        univariate_projection = config["selected_trials_spikes_fr_voltage"][trial_idx]["univariate_projection"]

        # Identifying the utilized_channels
        cur_template = extract_template(
            config, templateId=config["NEURON_ID"])
        utilized_channels = np.array(return_utilized_channels(cur_template))

        # Shape (num_time_bins for 6 seconds, num_utilized_channels)
        selected_univariate_projection = univariate_projection[:,
                                                               utilized_channels]

        positive_sum_per_trial = 0
        negative_sum_per_trial = 0
        for time_bin in selected_univariate_projection:
            # Iterating over each time bin, and calculating the summation of all the positive numbers
            for channel in time_bin:
                if channel > 0:
                    positive_sum_per_trial += channel
                else:
                    negative_sum_per_trial += channel
        result[trial_idx]['positive_sum'] = positive_sum_per_trial
        result[trial_idx]['negative_sum'] = negative_sum_per_trial

        summation_univariate_projection = np.sum(univariate_projection, axis=1)
        denoised_summation = pywt.threshold(summation_univariate_projection, value=np.std(
            summation_univariate_projection), mode='soft')
        result[trial_idx]['denoised_projection'] = denoised_summation
        result[trial_idx]['summation_univariate_projection'] = summation_univariate_projection

    return result


def plot_univariate_projection_stats(config):

    # List of channels
    univariate_projection_stats = config['univariate_projection_stats']
    trials = list(univariate_projection_stats.keys())
    positions = list(range(len(trials)))

    # Lists of metrics
    positive_sum = [univariate_projection_stats[trial]
                    ['positive_sum'] for trial in trials]
    negative_sum = [univariate_projection_stats[trial]
                    ['negative_sum'] for trial in trials]

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(2, figsize=(5, 10))

    # Plot the data
    axs[0].bar(positions, positive_sum, color='blue')
    axs[0].set_title('Positive Sum of all the positive projection values')
    axs[0].set_ylabel('Value')
    axs[0].set_xticks(positions)
    axs[0].set_xticklabels(trials, rotation='vertical')

    axs[1].bar(positions, negative_sum, color='orange')
    axs[1].set_title('Negative Sum of all the positive projection values')
    axs[1].set_ylabel('Value')
    axs[1].set_xticks(positions)
    axs[1].set_xticklabels(trials, rotation='vertical')

    fig.suptitle(f'{config["NEURON_ID"]} {config["desired_trial_type_name"]}')
    plt.show()


def heatmap_univariate(config):
    n = len(config["selected_trials"])  # number of subfigures
    fig, axs = plt.subplots(n, 1, figsize=(3, n*3))

    summation_heatmap_univariate = {}
    for ax, trial_idx in zip(axs, config["selected_trials"]):
        # shape should be (num_time_bins for 6 seconds, num_channels)
        univariate_projection = config["selected_trials_spikes_fr_voltage"][trial_idx]["univariate_projection"]

        # Identifying the utilized_channels
        cur_template = extract_template(config, templateId=config["NEURON_ID"])
        utilized_channels = np.array(return_utilized_channels(cur_template))
        univariate_projection = univariate_projection[:, utilized_channels].transpose(
        )

        summation_univariate_projection = np.sum(univariate_projection, axis=0)
        denoised_summation = pywt.threshold(summation_univariate_projection, value=np.std(
            summation_univariate_projection), mode='soft')
        summation_heatmap_univariate[trial_idx] = denoised_summation

        img = ax.imshow(univariate_projection,
                        interpolation='nearest', aspect='auto', cmap='seismic')

        fig.colorbar(img, ax=ax)
        ax.set_title(f'{trial_idx}')

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(len(summation_heatmap_univariate), 1, figsize=(
        5, len(summation_heatmap_univariate)*5))

    fig, axs = plt.subplots(n, 1, figsize=(5, n*5))

    for ax, trial_idx in zip(axs, summation_heatmap_univariate):
        ax.plot(summation_heatmap_univariate[trial_idx])
        ax.set_title(f'{trial_idx}')

    plt.tight_layout()
    plt.show()
