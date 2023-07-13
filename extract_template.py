import numpy as np
from display_spike_and_voltage import raw_spike_data_path, neuron_identity_data_path, trial_time_info_path, trialNum, TIME_BEFORE_CUE, TIME_AFTER_CUE, spike_templates_data_path, template_used_data_path, voltage_data_path
from spike_raster_display import extract_single_neuron_spike_times_for_specific_trial, center_around_cue, sliding_histogram, extract_processed_neuron_raster
from raw_voltage import extractPeakChannel, getDataAndPlot
from matplotlib import pyplot as plt
from scipy.signal import correlate
import os 
import scipy.stats
import time
import math



def return_utilized_channels(template):
    """Returns the channels that are utilized in the template (List)"""
    for temp in template:
        non_zero_indices = [i for i, num in enumerate(temp) if num != 0]
        if non_zero_indices:
            return non_zero_indices


def extract_template(template_used_data_path, templateId):
    """Extracts the template from the template_used_data_path (Should be something like channel counts (383) X timestamps (82))"""
    template = np.load(template_used_data_path)
    return template[templateId]


def multivariate(raw_voltage, template, utilized_channels):
    """
    This function calculates normalized multi-variate analysis of raw voltage data using a template.
    """
    totalNumBins = raw_voltage.shape[1]/template.shape[0] # should be number of samples / 82
    print("totalNumBins", totalNumBins)

    # should have N (number of bins) x 1 (dot product of x and y)
    final_Multivariate = [] 

    for bin in range(int(totalNumBins)):
        
        x = raw_voltage[utilized_channels, bin*template.shape[0]:(bin+1)*template.shape[0]].transpose().flatten()
        y = template[:,utilized_channels].flatten()
        #x and y must first be converted into 82 x 383 martix, select only the utilized Channels then flattened
        # print("x", x.shape, "y", y.shape)
        multivariate = np.dot(x, y)

        mag_x = np.linalg.norm(x)
        mag_y = np.linalg.norm(y)
        if mag_x != 0 and mag_y != 0:
            normalized_multivariate = multivariate/(mag_x*mag_y)
        else:
            normalized_multivariate = 0 
        
        final_Multivariate.append(normalized_multivariate)

    print("final_Multivariate", len(final_Multivariate))
    return np.array(final_Multivariate)
    
def univariate(raw_voltage, template):
    """
    This function calculates normalized univariate analysis of raw voltage data using a template.
    """
    totalNumBins = raw_voltage.shape[1]/template.shape[0] # should be number of samples / 82
    print("totalNumBins", totalNumBins)

    # should have N (number of bins) x 383 (number of channales)
    finalUnivariate = []

    #TODO: I should have only selected "utilized_channels" in the first place, but I did it later on in the code that it doesn't affect
    # The final output. But i should fix for clarity. 
    for bin in range(int(totalNumBins)):
        current_bin_prediction = []
        # current_bin_voltage should have shape 383 X 82
        current_bin_voltage = raw_voltage[:, bin*template.shape[0]:(bin+1)*template.shape[0]]
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

    print("finalUnivariate", len(finalUnivariate), len(finalUnivariate[0]))
    print('total Num Bins', totalNumBins)
    return np.array(finalUnivariate)

def plot_univariate(univariate_data, utilized_channels):
    """Plots the raw projection of the voltage data onto the template."""
    for channel in utilized_channels:
        plt.plot(univariate_data[:, channel], label='Channel ' + str(channel))

    plt.ylabel('Normalized Univariate')    
    plt.xlabel('#bins (each bin is 82 timepoints)')
    plt.legend()
    plt.show()

def convert_dot_product_to_spike_count_univariate(univariate_data, utilized_Channels, threshold):
    """Converts the projections of the voltage data onto the template into spike counts"""
    spike_count_per_bin = []
    for bin in range(univariate_data.shape[0]):
        #TODO: I originally did if np.mean(univariate_data[bin][utilized_Channels]) > threshold, and it performed much better than the current implementation
        # if (np.mean(univariate_data[bin][utilized_Channels]) > threshold):
        #     spike_count_per_bin.append(1)
        # else:
        #     spike_count_per_bin.append(0)
        if all(i > threshold for i in univariate_data[bin][utilized_Channels]):
            spike_count_per_bin.append(1)
            # print("bin", bin, "mean", np.mean(univariate_data[bin][utilized_Channels]))
        else:
            spike_count_per_bin.append(0)
    #returns a list of spike counts per bin, each count is separated out by 82 timepoints
    spike_count_per_bin = normalize_bin_count_to_spike_time(spike_count_per_bin)
    return spike_count_per_bin

def convert_dot_product_to_spike_count_multivariate(multivariate_projection, utilized_Channels, threshold):
    '''converts the projections of multi-varaite data onto the template into spike counts'''
    spike_count_per_bin = []

    for bin in range(multivariate_projection.shape[0]):
        if multivariate_projection[bin] > threshold:
            spike_count_per_bin.append(1)
        else:
            spike_count_per_bin.append(0)
    spike_count_per_bin = normalize_bin_count_to_spike_time(spike_count_per_bin)
    return spike_count_per_bin
    

def plot_uni_multivariate_spike_count_per_trial(spike_count_per_bin, NEURON_ID, trialNum, threshold, multivariate = False):
    """Takes a list of spike counts per timebin and plots it."""

    if len(spike_count_per_bin) == 0:
        if multivariate:
            print("spike_count_per_bin is empty for multivariate")
        else:
            print("spike_count_per_bin is empty for univariate ")
        return 
    # print("spike_count_per_bin", len(spike_count_per_bin))
    for spike_time in spike_count_per_bin:
        plt.axvline(x=spike_time, color='b')

    plt.axvline(x=0, color='r', label = 'Go Cue')
    plt.legend()
    plt.ylabel('Spike Count') 
    plt.xlabel('seconds')
    if multivariate:
        title = f'Multivariate Recovered NEURON: {NEURON_ID} Trial ' + str(trialNum) + f' threshold:{threshold}'
    else:
        title = f'Univariate Recovered NEURON: {NEURON_ID} Trial ' + str(trialNum) + f' threshold:{threshold}'
    plt.title(title)
    plt.show()

def compare_spike_count_similarity(computed_spike, kilosort_spike):
    #Need to convert computed spike into -3 and 3 timescale
    """This needs to be updated to use the new spike count per bin"""
    computed_spike = np.array(computed_spike)
    kilosort_spike = np.array(kilosort_spike)

    c = np.correlate(computed_spike, kilosort_spike)
    cross_corr = c / np.sqrt(np.sum(computed_spike ** 2) * np.sum(kilosort_spike ** 2))
    return cross_corr

def plot_recovered_spike_against_kilosort(computed_spike, kilosort_spike, neuron_id, trial_num, threshold, save_folder_path = None, multivariate = False):
    """Plots the recovered spike against the kilosort spike"""
    if len(computed_spike) == 0:
        if multivariate:
            print("computed_spike is empty for multivariate")
        else:
            print("computed_spike is empty for univariate ")
        return
    cross_corr = druckmann_correlation_calculation(computed_spike, kilosort_spike)
    fig, ax1 = plt.subplots()
    for i, spike in enumerate(computed_spike):
        if i == 0:
            if multivariate:
                ax1.vlines(spike, 0, 1, color='blue', label='Multi-variate recovered Spike')
            else:
                ax1.vlines(spike, 0, 1, color='blue', label='Uni-variate recovered Spike')
        else:
            ax1.vlines(spike, 0, 1, color='blue')
    for i, spike in enumerate(kilosort_spike):
        if i == 0:
            ax1.vlines(spike, 0, 1, color='red', label="kilosort computed spikes")
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

def normalize_bin_count_to_spike_time(spike_count_per_bin):
    """Takes a list of spike counts per bin (as indicated by 0 as no spikes during that bin, and 1 is yes spike during that bin), 
    and convert to a list of spike times."""
    time_arr_length = len(spike_count_per_bin) 
    time_bin_length_in_sec = 6/time_arr_length
    result_arr = [-3 + i*time_bin_length_in_sec for i, spike in enumerate(spike_count_per_bin) if spike==1 ]
    
    return result_arr

def convert_spike_time_to_fr(spikeTimes, begin_time, end_time, bin_width, stride, rate=True):
    #Convert 1D array to 3D Array
    spikeTimes = np.array(spikeTimes)
    
    bin_centers, bin_fr = sliding_histogram(spikeTimes, begin_time, end_time, bin_width, stride, rate=True)

    return bin_centers, bin_fr

def pearson_fr_calculation(computed_fr, kilosort_fr):
    return scipy.stats.pearsonr(computed_fr, kilosort_fr)[0]

def druckmann_correlation_calculation(computed_spike, kilosort_spike):
    #Both input should be the times of the spikes between -3 and +3 seconds
    precision_delta = 0.005 #5ms

    #Calculate the number of spikes that are within +- 2.5ms of each other (since we need to double 5ms)
    coincidences_count = 0
    # print("computed spike", len(computed_spike))
    # print("kilosort spike", len(kilosort_spike))
    # print("computed spike", computed_spike)
    # print("kilosort spike", kilosort_spike)

    for spike in computed_spike:
        for real_spike in kilosort_spike:
            if abs(spike - real_spike) <= precision_delta/2:
                coincidences_count += 1
    
    #Calculate the expected number of coincidences generated by a homogeneous poisson process with the same rate f as the spike train kilosort_spike
    real_fr = len(kilosort_spike)/6
    # probability_of_zero_spike_in_window = math.e ** (-1 * real_fr * precision_delta)
    # expected_coincidences = 1 - probability_of_zero_spike_in_window #This would include the case where there are 2 spikes, or 3 spikes, however, I would both count them as 1 coincidence
    expected_coincidences = 2 * real_fr * precision_delta * len(computed_spike)

    # print(f"coincidences_count: {coincidences_count}")
    # print(f"expected_coincidences: {expected_coincidences}")

    bottom = 0.5 * (len(computed_spike) + len(kilosort_spike))
    top = (coincidences_count - expected_coincidences)
    N = (1 - 2 * real_fr * precision_delta)
    # print(f"top: {top}")
    # print(f"bottom: {bottom}")
    # print(f"N: {N}")
    return top/(bottom * N)

def plot_fr_comparison(univariate_bin_fr, multivariate_bin_fr, kilosort_bin_fr, NEURON_ID, trialNum, threshold, save_folder_path = None):
    '''univariate_bin_fr, multivariate_bin_fr, and kilosort_bin_fr should have the same size (since they are both binned up for 6 seconds)'''
    x = np.arange(len(univariate_bin_fr))
    fig, axs = plt.subplots(3, 1, figsize=(15,10))

    # Plot for univariate_bin_fr
    title = f"Neuron_{NEURON_ID}_Trial_{trialNum}_Threshold {threshold}_fr_comparison"
    axs[0].set_title('univariate retrieved spikes' + title)
    axs[0].plot(x, univariate_bin_fr, color = "red")
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Firing rate')

    # Plot for multivariate_bin_fr
    axs[1].set_title('multivariate retrieved spikes' + title)
    axs[1].plot(x, multivariate_bin_fr, color = "green")
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Firing rate')

    # Plot for kilosort_bin_fr
    axs[2].set_title('kilosort retrieved spikes' + title)
    axs[2].plot(x, kilosort_bin_fr, color = "blue")
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Firing rate')

    plt.tight_layout()
    if save_folder_path:
        plt.savefig(os.path.join(save_folder_path, title + ".png"))

    plt.show()


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
    
def main():
    # time_for_all_trials = 4193 #seconds took to record all trials
    #sampling rate is 30000 Hz
    NEURON_ID = 110 #same as the template ID. 
    template_id = NEURON_ID
    peakChannel = extractPeakChannel(NEURON_ID)
    threshold = 0.06
    trialNum = 120
    bin_width = 0.01
    stride = 0.01
    save_folder_path = f"./neuron_{NEURON_ID}/"

    # list_of_threshold = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    # list_of_neuron_id = extract_processed_neuron_raster(raw_spike_data_path, neuron_identity_data_path).keys()
    # list_of_trial_num = [i for i in range(1, 300)]

    #Grab original kilosort retrieved spikes
    kilosort_computed_spike_time, cue_time = extract_single_neuron_spike_times_for_specific_trial(raw_spike_data_path, neuron_identity_data_path, trial_time_info_path, trialNum, NEURON_ID, TIME_BEFORE_CUE, TIME_AFTER_CUE)
    kilosort_computed_spike_time = center_around_cue(kilosort_computed_spike_time, cue_time)

    #Grab the template and raw_voltage data used for this neuron, for the specified trial
    current_neuron_template = extract_template(template_used_data_path, template_id)
    raw_voltage_data_per_neuron = getDataAndPlot(voltage_data_path, cue_time + TIME_BEFORE_CUE, cue_time + TIME_AFTER_CUE, peakChannel, plot=False, allChannels=True)
    utilized_Channels = return_utilized_channels(current_neuron_template)

    #Convert the raw_voltage data into a univariate projection
    univariate_projection = univariate(raw_voltage_data_per_neuron, current_neuron_template)
    print("univariate_projection", univariate_projection)

    #Convert the raw_voltage data into a multivariate projection
    multivariate_projection = multivariate(raw_voltage_data_per_neuron, current_neuron_template, utilized_Channels)
    print("multivariate_projection", multivariate_projection)

    #Convert the projection into spikes
    univariate_recovered_spike = convert_dot_product_to_spike_count_univariate(univariate_projection, utilized_Channels, threshold)
    multivariate_recovered_spike = convert_dot_product_to_spike_count_multivariate(multivariate_projection, utilized_Channels, threshold)

     
    print("kilosort_computed_spike_time", len(kilosort_computed_spike_time), kilosort_computed_spike_time)
    print("univariate_recovered_spike", len(univariate_recovered_spike), univariate_recovered_spike)
    print("multivariate_recovered_spike", len(multivariate_recovered_spike), multivariate_recovered_spike)
    
    # Plot the spikes individually then against the kilosort recovered spikes 
    plot_uni_multivariate_spike_count_per_trial(univariate_recovered_spike, NEURON_ID, trialNum, threshold)
    plot_uni_multivariate_spike_count_per_trial(multivariate_recovered_spike, NEURON_ID, trialNum, threshold, multivariate=True)

    plot_recovered_spike_against_kilosort(univariate_recovered_spike, kilosort_computed_spike_time, NEURON_ID, trialNum, threshold, save_folder_path)
    plot_recovered_spike_against_kilosort(multivariate_recovered_spike, kilosort_computed_spike_time, NEURON_ID, trialNum, threshold, save_folder_path, multivariate=True)

    #Since we're using a bin width of 0.01, and some of the neurons fire sporadically, the firing rate returned is in Hz, and it might be 100Hz, which is higher than the real firing rate
    
    #Convert the spike count into firing rate
    bin_centers, univariate_bin_fr = convert_spike_time_to_fr(univariate_recovered_spike, TIME_BEFORE_CUE, TIME_AFTER_CUE, bin_width, stride, rate=True)
    _, multivariate_bin_fr = convert_spike_time_to_fr(multivariate_recovered_spike, TIME_BEFORE_CUE, TIME_AFTER_CUE, bin_width, stride, rate=True)
    _, kilosort_bin_fr = convert_spike_time_to_fr(kilosort_computed_spike_time, TIME_BEFORE_CUE, TIME_AFTER_CUE, bin_width, stride, rate=True)

    plot_fr_comparison(univariate_bin_fr, multivariate_bin_fr, kilosort_bin_fr, NEURON_ID, trialNum, threshold, save_folder_path)

    print("bin_centers", bin_centers)
    print("bin_fr", univariate_bin_fr)
    print("kilosort_bin_fr", kilosort_bin_fr)
    print('Univariate Pearson correlation coefficient: ', scipy.stats.pearsonr(univariate_bin_fr, kilosort_bin_fr)[0])
    print('Multivariate Pearson correlation coefficient: ', scipy.stats.pearsonr(multivariate_bin_fr, kilosort_bin_fr)[0])
    print('univariate kilosort correlation druckmann equation:', druckmann_correlation_calculation(univariate_recovered_spike, kilosort_computed_spike_time))
    print('multivariate kilosort correlation druckmann equation:', druckmann_correlation_calculation(multivariate_recovered_spike, kilosort_computed_spike_time))



# threshold_results, pearson_corr_threshold = find_optimal_threshold(list_of_threshold, list_of_neuron_id, list_of_trial_num, raw_spike_data_path, neuron_identity_data_path, trial_time_info_path, voltage_data_path, template_used_data_path, TIME_BEFORE_CUE, TIME_AFTER_CUE, bin_width, stride)

# print("threshold_results", threshold_results)
# print("pearson_corr_threshold", pearson_corr_threshold)

# np.save("./threshold_results.npy", threshold_results)
# np.save("./pearson_corr_threshold.npy", pearson_corr_threshold)

if __name__ == "__main__":
    main()