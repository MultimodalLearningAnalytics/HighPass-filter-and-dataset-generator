import pandas as pd
from datetime import datetime
import scipy
import sys
from scipy import signal
from scipy import pi
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz


# ----- ----- ----- -----
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


# ----- -----
# (1)
def foo(sel, datastream):
    if sel == 1:
        headers = ['Date', 'Unfiltered1', 'Filtered1', 'Unfiltered2', 'Filtered2']
        df = pd.read_csv('FilteredUnfiltered.CSV', names=headers)
        print(df)

        df['Date'] = df['Date'].map(lambda x: datetime.strptime(str(x), '%Y/%m/%d %H:%M:%S.%f'))
        x = df['Date']
        y1 = df['Unfiltered1']
        y2 = df['Filtered1']
        y3 = df['Unfiltered2']
        y4 = df['Filtered2']

        # Filter requirements.
        order = 6
        fs = 100.0  # sample rate, Hz
        cutoff = 0.2  # desired cutoff frequency of the filter, Hz

        # Get the filter coefficients so we can check its frequency response.
        b, a = butter_highpass(cutoff, fs, order)

        # Plot the frequency response.
        # w, h = freqz(b, a, worN=8000)
        # plt.subplot(2, 1, 1)
        # plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
        # plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
        # plt.axvline(cutoff, color='k')
        # # plt.xlim(0, 0.5 * fs)
        # plt.xlim(0, cutoff*3)
        # plt.title("High Filter Frequency Response")
        # plt.xlabel('Frequency [Hz]')
        # plt.grid()

        # Demonstrate the use of the filter.
        # First make some data to be filtered.
        T = 5  # seconds
        # n = int(T * fs)  # total number of samples
        # n = 8121
        n = 400
        t = np.linspace(0, T, n, endpoint=False)
        # "Noisy" data.  We want to recover the 20 Hz signal from this.
        data = np.sin(1.2 * 2 * np.pi * t) + 1.5 * np.cos(5 * 2 * np.pi * t) + 0.5 * np.sin(20.0 * 2 * np.pi * t)

        # Filter the data, and plot both the original and filtered signals.
        # y = butter_highpass_filter(data, cutoff, fs, order)
        y = butter_highpass_filter(datastream, cutoff, fs, order)
        print(y)

        # plt.subplot(2, 1, 2)
        # # plt.plot(t, data, 'b-', label='data')
        # plt.plot(t, datastream, 'b-', label='data')
        # plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
        # plt.xlabel('Time [sec]')
        # plt.grid()
        # plt.legend()
        #
        # plt.subplots_adjust(hspace=0.35)
        # plt.show()

        return y
    else:
        print ('Please, choose among choices, thanks.')


# ----- -----
def main():
    # sel = int (sys.argv[1])
    experiment = 3
    participant = 1



    if experiment == 2:
        print("Experiment 2")
    elif experiment == 3:
        print("Experiment 3")
        path = "C://Users/giuse/Desktop/OutputFiles/Experiment" + str(experiment) + "/participant" + str(participant) + "/phoneAcceleromenterDistractionExp" + str(experiment) + "Part" + str(participant) + ".csv"

        first = pd.read_csv(path, header=None)
        print(first)

        first[1] = 5 - first[1]
        first = first.drop([5], axis=1).drop([0], axis=1).to_numpy()
        # print(first)
        # first[0] = 5 - first[0]
        print(first)

        prev = 5.1
        final_list = []
        datas = []
        count = 0
        min = 500
        for row in first:
            if row[0] > prev:
                datas = np.array(datas)
                lol = np.array([datas, 1])
                # print(datas.shape)
                rows, cols = datas.shape
                datas = datas[rows - 400:]
                print(datas.shape)
                min = rows if rows < min else min
                # final_list.append(lol)
                final_list.append(datas)
                datas = []
                count += rows
            datas.append(row)
            prev = row[0]

        datas = np.array(datas)
        lol = np.array([datas, 1])
        # print(datas.shape)
        rows, cols = datas.shape
        datas = datas[rows - 400:]
        print(datas.shape)
        min = rows if rows < min else min
        # final_list.append(lol)
        final_list.append(datas)
        datas = []
        count += rows
        print ("Final count: " + str(final_list.__len__()))
        print(count)
        print(min)
        transpose = np.transpose(final_list[0])
        print(transpose)
        print(transpose[1])

        print(foo(1, transpose[1]))
        newtranspose = []
        newtranspose.append(transpose[0])
        newtranspose.append(foo(1, transpose[1]))
        newtranspose.append(foo(1, transpose[2]))
        newtranspose.append(foo(1, transpose[3]))

        np.savetxt("dataoutput/Exp" + str(experiment) + "Part" + str (participant) + ".txt", newtranspose, delimiter=" ")



# ----- ----- ----- ----- ----- -----
if __name__ == '__main__':
    main()