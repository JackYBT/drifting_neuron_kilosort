import numpy as np
import scipy.io as sio
import os
import hdf5storage
import matplotlib.pyplot as plt
import math


def extract_processed_neuron_raster(raw_spike_data_path, neuron_identity_data_path):
    """
    This function extracts and processes spike raster data of neurons from raw spike data and neuron's identity data.

    Args:
    raw_spike_data_path: str, path to the file that contains the spike times in seconds for all neurons across all trials.
    neuron_identity_data_path: str, path to the file that contains the neuron identity for each spike time

    The function loads data from numpy files, then organizes spike times for each neuron in a dictionary.

    Returns:
    spike_times: dict, a dictionary where keys are neuronId and values are spike times for the specific neuron.
    """

    # Load the raw spike data from numpy file
    data = np.load(raw_spike_data_path)
    # Has shape (n, 1) where n is the number of spikes

    # Load the neuron identity data from numpy file
    spike_clusters_neuron_identity = np.load(neuron_identity_data_path)
    # Has shape (n, 1) where n is the number of spikes, 0-based, so there is a 0 neuronId

    # Initialize an empty dictionary to store spike times for each neuron, and the template that was used for that specific spike

    spike_times = {}

    # print("data.shape: ", data.shape)
    # print("spike neuron identity.shape", spike_clusters_neuron_identity.shape)
    # print("min(spike_clusters_neuron_identity): ", min(spike_clusters_neuron_identity))
    # print("max(spike_clusters_neuron_identity): ", max(spike_clusters_neuron_identity))
    # Iterate over each spike time
    for idx, spike_time in enumerate(data):
        # If the neuronId doesn't exist in the dictionary, add a new list
        if spike_clusters_neuron_identity[idx] not in spike_times:
            spike_times[spike_clusters_neuron_identity[idx]] = []

        # Add the spike time to the neuronId's list
        spike_times[spike_clusters_neuron_identity[idx]].append(spike_time)

    # Return the dictionary containing spike times for each neuron
    return spike_times


def extract_single_neuron_spike_times_for_specific_trial(raw_spike_data_path, neuron_identity_data_path, trial_time_info_path, trialNum, neuronId, TIME_BEFORE_CUE=-3, TIME_AFTER_CUE=3):
    """
    This function extracts the spike times for a specific neuron for a specific trial.

    Args:
    spike_times: dict, a dictionary where keys are neuronId and values are spike times for the specific neuron.
    cue_time: list, a list of cue times for all trials
    neuronId: int, neuron identity, 0 based
    trialNum: int, trial number, 0 based

    The function extracts the spike times for the specified neuron for the specified trial.

    Returns:
    adjustedNeuronSpikeTime: list, a list of spike times for the specified neuron for the specified trial
    """
    _, cue_time = extract_trial_time_info(trial_time_info_path)

    adjustedNeuronSpikeTime = extract_single_neuron_spike_times_between_specific_times(
        raw_spike_data_path, neuron_identity_data_path, cue_time[trialNum] + TIME_BEFORE_CUE, cue_time[trialNum] + TIME_AFTER_CUE, neuronId)

    # print("adjustedNeuronSpikeTime (6 seconds around cue time): ", adjustedNeuronSpikeTime)
    # print("*"*100)
    # A list of times of the specified neuron firing time for the specified trial, which is defined as around 3 seconds of the cue time
    return adjustedNeuronSpikeTime, cue_time[trialNum]


def extract_single_neuron_spike_times_between_specific_times(raw_spike_data_path, neuron_identity_data_path, t0, t1, neuronId):
    # raw_spike_data_path: str, path to the file that contains the spike times in seconds for all neurons across all trials.
    # neuron_identity_data_path: str, path to the file that contains the neuron identity for each spike time
    # t0: float, beginning time of the desired neuron spiking rate
    # t1: float, end time of the desired neuron spiking rate
    # neuronId: int, neuron identity

    # Load the raw spike data from numpy file
    spike_times = extract_processed_neuron_raster(
        raw_spike_data_path, neuron_identity_data_path)
    unadjustedNeuronSpikeTime = spike_times[neuronId]
    # Extract the spike times for the desired neuron

    # Do not center the spike time around anything, keep the spike time the way it is
    # Only the spikes that fall within the t0 and t1 are kept for the desired neuron.
    adjustedNeuronSpikeTime = [
        spike_time for spike_time in unadjustedNeuronSpikeTime if t0 < spike_time < t1]

    # A list of times of the specified neuron firing time between t0 and t1
    return adjustedNeuronSpikeTime


def extract_trial_time_info(mat_file_path):
    """
    Function to extract trial timing info from .mat file.

    Parameters:
    mat_file_path (str): String representing the file path of the .mat file.

    Returns:
    tuple: tuple containing the beginning trial time and cue time.

    Raises:
    ValueError: if there is a problem with loading the .mat file.
    Exception: if any other unknown fault occurs.
    """

    try:
        # Set the path for the .mat file
        # Check if the file exists at the given path
        if os.path.isfile(mat_file_path):
            # Load the .mat file
            trial_times = hdf5storage.loadmat(mat_file_path)
        else:
            print(f"No such file found at path: {mat_file_path}")
    except ValueError as e:
        print(f"An error occurred while trying to load a .mat file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Extract the beginning trial time from the .mat file
    beginning_trial_time = trial_times['onset'][0][0][0][0]
    # Extract the end trial time from the .mat file
    end_trial_time = trial_times['offset'][0][0][0][0]
    # Extract the cue time from the .mat file
    cue_time = trial_times['onset'][0][0][2][0]

    # Return the beginning trial time and the cue time as a tuple
    return beginning_trial_time, cue_time


def upscale_list(numbers, cue, scale_factor):
    return [(num - cue) * scale_factor for num in numbers]


def plot_single_neuron_spikes_between_times(spike_time, neuronId, t0, t1, trialNum=None):
    print("spike_time", spike_time)
    print("T0", t0)
    print("t1", t1)
    cue = (t0 + t1)/2

    assert np.all((t0 < np.asarray(spike_time)) & (np.asarray(
        spike_time) < t1)), "All spike times should be between t0 and t1"
    # spike_time = upscale_list(spike_time, 1000) - cue*1000 # Convert spike time from seconds to milliseconds, and center around cue
    spike_time = upscale_list(spike_time, cue, 1000)
    plt.vlines(spike_time, 0, 1, colors='k')
    plt.axvline(x=0, color='r', label='Go Cue')
    plt.legend()
    plt.xlabel('Spike time (ms)')
    # Set the y-axis label
    plt.ylabel('Spike')
    # Set the plot title
    plt.title(
        f'Neuron {neuronId} Raster Plot{" for Trial " + str(trialNum) if trialNum else ""}')
    # Display the plot
    plt.show()


def plot_single_neuron_spikes_across_all_trials(spike_time_across_trials, neuronId):
    """
    This function is designed to generate a raster plot for a single neuron's spike times across all trials. 
    Each trial is represented as a horizontal line segment, with each spike represented as a vertical line.

    Parameters:
    spike_time_across_trials (list): A list of arrays where each array contains the spike times for one trial, however, it would also work if it is one single element
    neuronId (int): An integer identifier for the neuron.

    Returns:
    None. Plot is displayed using plt.show() within the function itself.
    """
    # Create a new figure with specified size
    plt.figure(figsize=(10, 7))

    # For each individual trial
    for i, trial in enumerate(spike_time_across_trials):
        # Create a vertical line for each spike time in the trial
        plt.vlines(trial, i + .5, i + 1.5, colors='k')

    # Set the y-axis limits
    plt.ylim(.5, len(spike_time_across_trials) + .5)
    # Set the x-axis label
    plt.xlabel('Spike time')
    # Set the y-axis label
    plt.ylabel('Trial')
    # Set the plot title
    plt.title(f'Neuron {neuronId} Raster Plot')
    # Display the plot
    plt.show()


def extract_single_neuron_spike_time_across_trials(spike_times, cue_time, neuronId, TIME_BEFORE_CUE=-3, TIME_AFTER_CUE=3):
    """
    This function adjusts the spiking times of a neuron during a number of trials to be centered around 
    a given cue time. Only the spikes that fall in a predefined time window around the cue are kept.

    Parameters:
    spike_times (list of lists of floats): Each list corresponds to one neuron and contains the spiking times of that neuron.
    cue_time (list of floats): List containing the cue times. The length equals the number of trials.
    neuronId (int): The index of the neuron to process in the spike_times list.

    Returns:
    adjustedNeuronSpikeTime (list of lists of floats): Each inner list corresponds to one trial,
    and contains the spiking times (between -3 and 3 seconds relative to the cue time) of the selected neuron.
    """
    # print("spike_times", sorted(spike_times.keys()))
    if neuronId not in spike_times.keys():
        print(f"Neuron {neuronId} not found in spike_times")
        return None
    unadjustedNeuronSpikeTime = spike_times[neuronId]

    # For each trial, adjust the spike times to be centered around the cue time by subtracting the cue time.
    # Only the spikes that fall within the BEGIN_COUNT and END_COUNT window are kept.
    adjustedNeuronSpikeTime = [
        [
            spike_time - cue
            for spike_time in unadjustedNeuronSpikeTime
            if cue + TIME_BEFORE_CUE < spike_time < cue + TIME_AFTER_CUE
        ]
        for cue in cue_time
    ]

    return adjustedNeuronSpikeTime


def center_around_cue(arr, cue):
    return arr - cue


def sliding_histogram(spikeTimes, begin_time, end_time, bin_width, stride, rate=True):
    '''
    Calculates the number of spikes for each unit in each sliding bin of width bin_width, strided by stride.
    begin_time and end_time are treated as the lower- and upper- bounds for bin centers.
    if rate=True, calculate firing rates instead of bin spikes.

    Parameters:

    spikeTimes (ndarray): 1D NumPy array of spike times.
    begin_time (float): Start of the time range.
    end_time (float): End of the time range.
    bin_width (float): Width of each slide window/bin.
    stride (float): Stride length for sliding the bins.
    rate (bool): Flag to indicate whether to calculate firing rates instead of bin spikes. Defaults to True. 

    Returns:
    binCenters (np.array): Calculated center of each bin.
    binSpikes (np.array): Calculated spike counts or firing rates (depending on 'rate' parameter) for each bin.
    '''
    bin_begin_time = begin_time
    bin_end_time = end_time
    # This is to deal with cases where for e.g. (bin_end_time-bin_begin_time)/stride is actually 43 but evaluates to 42.999999... due to numerical errors in floating-point arithmetic.
    if np.allclose((bin_end_time-bin_begin_time)/stride, math.floor((bin_end_time-bin_begin_time)/stride)+1):
        n_bins = math.floor((bin_end_time-bin_begin_time)/stride)+2
    else:
        n_bins = math.floor((bin_end_time-bin_begin_time)/stride)+1
    binCenters = bin_begin_time + np.arange(n_bins)*stride

    binIntervals = np.vstack(
        (binCenters-bin_width/2, binCenters+bin_width/2)).T

    # Count number of spikes for each sliding bin
    binSpikes = np.asarray([np.sum(np.all(
        [spikeTimes >= binInt[0], spikeTimes < binInt[1]], axis=0)) for binInt in binIntervals])

    if rate:
        return binCenters, binSpikes/float(bin_width)  # get firing rates
    return binCenters, binSpikes
