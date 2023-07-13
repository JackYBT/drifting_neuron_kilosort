import pandas as pd
import numpy as np
import os

from raw_voltage_helpers import *

#365,complete_session,217,3.748408074946122,0.6318257956448912,0.3296482412060302,0.3163613029160173,0.22574243375185418,-0.024029346793889073,108.99356122432809,165.0,0.0784876764776263,0.6540639706468853

def extractPeakChannel(neuronId, filename = "imec1_ks2/waveform_metrics.csv"):
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        df = df[df['cluster_id'] == neuronId]
        return df['peak_channel'].values[0]
    else:
        print(f"No such file found at path: {filename}")
        return None

def getMaxChannelNumber(filename = "imec1_ks2/waveform_metrics.csv"):
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        return np.max(df['peak_channel'].values)
    else:
        print(f"No such file found at path: {filename}")
        return None


def getDataAndPlot(file_path, tStart, tEnd, peakChannel, trialNum = None, dataType='A', dw=0, dLineList=[0, 1, 6], plot=True, allChannels=False):
    #tStart, tEnd is in seconds
    binFullPath = Path(file_path)

    # Other parameters about what data to read
    dataType = 'A'    # 'A' for analog, 'D' for digital data

    # For analog channels: zero-based index of a channel to extract,
    # gain correct and plot (plots first channel only)
    if not allChannels:
        chanList = [peakChannel + i for i in range(-5,6)]
    else:
        chanList = list(range(383))


    # For a digital channel: zero based index of the digital word in
    # the saved file. For imec data there is never more than one digital word.
    dw = 0

    # Zero-based Line indices to read from the digital word and plot.
    # For 3B2 imec data: the sync pulse is stored in line 6.
    dLineList = [0, 1, 6]

    # Read in metadata; returns a dictionary with string for values
    meta = readMeta(binFullPath)

    # parameters common to NI and imec data
    sRate = SampRate(meta)
    firstSamp = int(sRate*tStart)
    lastSamp = int(sRate*tEnd)
    # array of times for plot
    tDat = np.arange(firstSamp, lastSamp+1)
    tDat = 1000*tDat/sRate      # plot time axis in msec

    tDat = tDat - tDat[0] - 3000       # set t=-3 at start of plot

    rawData = makeMemMapRaw(binFullPath, meta)

    if dataType == 'A':
        #I think the second index is time
        selectData = rawData[chanList, firstSamp:lastSamp+1]
        # print("raw_data.shape, which is total number of samples", rawData.shape[1])
        # print("sampling rate", sRate)
        if meta['typeThis'] == 'imec':
            # apply gain correction and convert to uV
            print("unit is uV")
            convData = 1e6*GainCorrectIM(selectData, chanList, meta)
        else:
            MN, MA, XA, DW = ChannelCountsNI(meta)
            print("NI channel counts: %d, %d, %d, %d" % (MN, MA, XA, DW))
            print("unit is mV")
            # apply gain correction and convert to mV
            convData = 1e3*GainCorrectNI(selectData, chanList, meta)

        # Plot the first of the extracted channels
        print("ConvData", convData.shape)
        if not plot:
            return convData
        fig, axs = plt.subplots(4, 3, figsize=(15,10))

        for i in range(11):
            row = i // 3
            col = i % 3
            axs[row, col].set_title(f"Channel Number: {chanList[i]}")
            axs[row, col].plot(tDat, convData[i, :])
            axs[row, col].set_xlabel('(ms)')
            axs[row, col].set_ylabel('(uV)')
    
        axs[3,2].set_title(r"$\bf{Trial\ Number:}$" + str(trialNum) if trialNum else "")
        print("trialnum", trialNum)
        plt.tight_layout()

        plt.show()
    else:
        digArray = ExtractDigital(rawData, firstSamp, lastSamp, dw,
                                  dLineList, meta)

        # Plot the first of the extracted channels
        fig, ax = plt.subplots()

        for i in range(0, len(dLineList)):
           ax.plot(tDat, digArray[i, :])
        plt.show()

def main():
    BEGIN_TIME = 0
    END_TIME = 10
    peakChannel = extractPeakChannel(365, "imec1_ks2/waveform_metrics.csv")
    getDataAndPlot("./NL_NL106_20221103_session1_g0_tcat.imec1.ap.bin", BEGIN_TIME, END_TIME, peakChannel)

