from spike_extraction_helpers import *
from raw_voltage import *


# displays the voltage and spikes for a given neuron between two times
def display_voltage_and_spikes_across_given_time(voltage_data_path, raw_spike_data_path, neuron_identity_data_path, neuron_channel_path, t0, t1, neuronId):
    singleNeuronSpike = extract_single_neuron_spike_times_between_specific_times(
        raw_spike_data_path, neuron_identity_data_path, t0, t1, neuronId)
    print("singleNeuronSpike: ", singleNeuronSpike)
    plot_single_neuron_spikes_between_times(
        singleNeuronSpike, neuronId, t0, t1)
    # A list of times of the specified neuron firing time between t0 and t1

    peakChannel = extractPeakChannel(neuronId, neuron_channel_path)
    getDataAndPlot(voltage_data_path, t0, t1, peakChannel)


# displays the voltage and spikes for a given trial and neuron
def display_voltage_and_spikes_given_trial(voltage_data_path, raw_spike_data_path, neuron_identity_data_path, neuron_channel_path, trial_time_info_path, trialNum, neuronId, TIME_BEFORE_CUE=-3, TIME_AFTER_CUE=3):
    singleNeuronSpikeSpecificTrial, cue_time = extract_single_neuron_spike_times_for_specific_trial(
        raw_spike_data_path, neuron_identity_data_path, trial_time_info_path, trialNum, neuronId, TIME_BEFORE_CUE, TIME_AFTER_CUE)
    print("cue_time: ", cue_time)
    plot_single_neuron_spikes_between_times(
        singleNeuronSpikeSpecificTrial, neuronId, cue_time + TIME_BEFORE_CUE, cue_time + TIME_AFTER_CUE, trialNum)

    peakChannel = extractPeakChannel(neuronId, neuron_channel_path)
    getDataAndPlot(voltage_data_path, cue_time + TIME_BEFORE_CUE,
                   cue_time + TIME_AFTER_CUE, peakChannel, trialNum)

# displays a given neuron's spikes across all trials


def display_spikes_across_all_trials(raw_spike_data_path, neuron_identity_data_path, trial_time_info_path, neuronId):
    spike_times = extract_processed_neuron_raster(
        raw_spike_data_path, neuron_identity_data_path)
    beginning_trial_time, cue_time = extract_trial_time_info(
        trial_time_info_path)

    for neuron in neuronId:
        adjustedSingleNeuronSpikeAcrossAllTrials = extract_single_neuron_spike_time_across_trials(
            spike_times, cue_time, neuron)
        if adjustedSingleNeuronSpikeAcrossAllTrials is None:
            continue
        plot_single_neuron_spikes_across_all_trials(
            adjustedSingleNeuronSpikeAcrossAllTrials, neuron)


def display_spikes_across_given_trials(listOfNeuronId):
    display_spikes_across_all_trials(
        raw_spike_data_path, neuron_identity_data_path, trial_time_info_path, listOfNeuronId)

# raw_spike_data_path = 'imec1_ks2/spike_times_sec_adj.npy'
# neuron_identity_data_path = 'imec1_ks2/spike_clusters.npy'
# voltage_data_path = "./NL_NL106_20221103_session1_g0_tcat.imec1.ap.bin"
# neuron_channel_path = "imec1_ks2/waveform_metrics.csv"
# trial_time_info_path = 'AccessarySignalTime.mat'
# # spike_templates_data_path = 'imec1_ks2/spike_templates.npy'
# template_used_data_path = 'imec1_ks2/templates.npy'
# trial_type_path = './NL106_yes_no_multipole_delay_stimDelay_Nov03_2022_Session1.mat'

# neuronId = 365
# t0 = 460 #seconds
# t1 = 470
# trialNum = 75
# TIME_BEFORE_CUE = -3
# TIME_AFTER_CUE = 3


def main():

    # This plots the desires neuron's across all trials neuron raster
    listOfNeuronId = list(range(872, 950))
    display_spikes_across_given_trials(listOfNeuronId)

    # display_voltage_and_spikes_across_given_time(voltage_data_path, raw_spike_data_path, neuron_identity_data_path, neuron_channel_path, t0, t1, neuronId)
    display_voltage_and_spikes_given_trial(voltage_data_path, raw_spike_data_path, neuron_identity_data_path,
                                           neuron_channel_path, trial_time_info_path, trialNum, neuronId, TIME_BEFORE_CUE, TIME_AFTER_CUE)


if __name__ == "__main__":
    main()
