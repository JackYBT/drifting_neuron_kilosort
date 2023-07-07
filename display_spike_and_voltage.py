from spike_raster_display import *
from raw_voltage import *

def main():
    raw_spike_data_path = 'imec2_ks2/spike_times_sec_adj.npy'
    neuron_identity_data_path = 'imec2_ks2/spike_clusters.npy'
    spike_times = extract_processed_neuron_raster(raw_spike_data_path, neuron_identity_data_path)
    #keys: neuronId, values: spike times for that specific neuron
    # But this is across all trials, so we need to extract info for a specific trial next

    trial_time_info_path = 'AccessarySignalTime.mat'
    beginning_trial_time, cue_time = extract_trial_time_info(trial_time_info_path)
    #Identifying all the beginning and end times of trials

    for neuronId in list(range(354, 370)):
        print("neuronId: ", neuronId)
        adjustedNeuronSpikeTime = extract_single_neuron_spiking_rate_across_trials(spike_times, cue_time, neuronId)
        plot_single_neuron_across_all_trials(adjustedNeuronSpikeTime, neuronId)

main()