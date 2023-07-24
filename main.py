from plot_helpers import extractPeakChannel, generate_desired_trials, compare_univariate_template, plot_spike_triggered_average, plot_elephant_correlation
from uni_multi_variate_helpers import get_uni_multi_kilosort_spikes_across_selected_trials, get_univariate_projection_stats, extract_all_templates
from model import train_model
from spike_extraction_helpers import extract_trial_time_info
import numpy as np


def main():

    # open a certain numpy file
    # amplitudes = np.load('imec1_ks2/amplitudes.npy')
    # for elem in amplitudes:
    #     print(elem)
    config = {
        'NEURON_ID': 274,
        'threshold': 0.11,
        'bin_width': 0.01,
        'stride': 0.01,
        'save_folder_path': "./neuron_365/",
        'selected_trials': [4, 10, 30],
        'amplitudes_data_path': 'imec1_ks2/amplitudes.npy',
        'raw_spike_data_path': 'imec1_ks2/spike_times_sec_adj.npy',
        'neuron_identity_data_path': 'imec1_ks2/spike_clusters.npy',
        'raw_voltage_data_path': "./NL_NL106_20221103_session1_g0_tcat.imec1.ap.bin",
        'neuron_channel_path': "imec1_ks2/waveform_metrics.csv",
        'trial_time_info_path': 'AccessarySignalTime.mat',
        'template_used_data_path': 'imec1_ks2/templates.npy',
        'templates_used_per_spike_data_path': 'imec1_ks2/spike_templates.npy',
        'optogenetic_stim_type_path': 'stim-trial-info.pkl',
        'trial_type_path': './NL106_yes_no_multipole_delay_stimDelay_Nov03_2022_Session1.mat',
        'TIME_BEFORE_CUE': -3,
        'TIME_AFTER_CUE': 3,
        'trialNum': 152,
        'optogenetic_desired_trial_type': 0,  # no optogenetic stimulation
        'lick_direction_desired_trial_type': 0,  # 0 left lick, 1 right lick
        'random_sample_number': 10,
        'begin_non_drift_trial_id': 0,
        'end_non_drift_trial_id': 150,
    }
    config['desired_trial_type_name'] = f'optogenetic stimulation type (f{config["optogenetic_desired_trial_type"]}) + lick direction ({config["lick_direction_desired_trial_type"]}) threshold {config["threshold"]}'
    config['template_id'] = config['NEURON_ID']
    config['peakChannel'] = extractPeakChannel(config['NEURON_ID'])
    config['all_templates'] = extract_all_templates(config)
    config['selected_trials'] = generate_desired_trials(config)
    # additional_trials = list(range(200, 210))
    # config['selected_trials'] = np.sort(np.concatenate(
    #     (config['selected_trials'], additional_trials)))  # add 10 more trials that drifted
    print(len(config['selected_trials']), config['selected_trials'])

    # print("all_cue_time.shape", len(all_cue_time), all_cue_time)
    config["selected_trials_spikes_fr_voltage"] = get_uni_multi_kilosort_spikes_across_selected_trials(
        config)
    # plot_elephant_correlation(config)
    config["univariate_projection_stats"] = get_univariate_projection_stats(
        config)
    compare_univariate_template(config)
    plot_spike_triggered_average(config)
    # train_model(config)


if __name__ == "__main__":
    main()
