import numpy as np
import scipy.io as sio
import os
import hdf5storage
import matplotlib.pyplot as plt


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
    data = np.load('imec2_ks2/spike_times_sec_adj.npy')

    # Load the neuron identity data from numpy file
    spike_clusters_neuron_identity = np.load('imec2_ks2/spike_clusters.npy')

    # Initialize an empty dictionary to store spike times for each neuron
    spike_times = {}

    # Iterate over each spike time
    for idx, spike_time in enumerate(data):
        # If the neuronId doesn't exist in the dictionary, add a new list 
        if spike_clusters_neuron_identity[idx] not in spike_times:
            spike_times[spike_clusters_neuron_identity[idx]] = []
        
        # Add the spike time to the neuronId's list
        spike_times[spike_clusters_neuron_identity[idx]].append(spike_time)

    # Return the dictionary containing spike times for each neuron
    return spike_times





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
        mat_file_path = 'AccessarySignalTime.mat'
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
    return beginning_trial_time,cue_time


def plot_single_neuron_across_all_trials(spike_time_across_trials, neuronId):
    """
    This function is designed to generate a raster plot for a single neuron's spike times across all trials. 
    Each trial is represented as a horizontal line segment, with each spike represented as a vertical line.
    
    Parameters:
    spike_time_across_trials (list): A list of arrays where each array contains the spike times for one trial.
    neuronId (int): An integer identifier for the neuron.
    
    Returns:
    None. Plot is displayed using plt.show() within the function itself.
    """
    # Create a new figure with specified size
    plt.figure(figsize=(10,7))
    
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



def extract_single_neuron_spiking_rate_across_trials(spike_times, cue_time, neuronId):
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
    
    unadjustedNeuronSpikeTime = spike_times[neuronId]

    # Define the time boundaries relative to the cue, to select the spikes
    BEGIN_COUNT, END_COUNT = -3, 3

    # For each trial, adjust the spike times to be centered around the cue time by subtracting the cue time.
    # Only the spikes that fall within the BEGIN_COUNT and END_COUNT window are kept.
    adjustedNeuronSpikeTime = [
        [
            spike_time - cue 
            for spike_time in unadjustedNeuronSpikeTime 
            if BEGIN_COUNT < spike_time - cue < END_COUNT
        ]
        for cue in cue_time
    ]
                
    return adjustedNeuronSpikeTime


# print(all_trials)
