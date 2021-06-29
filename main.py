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
def foo(sel, datastream, plot = False, T=5, n=120):
    if sel == 1:
        # headers = ['Date', 'Unfiltered1', 'Filtered1', 'Unfiltered2', 'Filtered2']
        # df = pd.read_csv('FilteredUnfiltered.CSV', names=headers)
        # print(df)
        #
        # df['Date'] = df['Date'].map(lambda x: datetime.strptime(str(x), '%Y/%m/%d %H:%M:%S.%f'))
        # x = df['Date']
        # y1 = df['Unfiltered1']
        # y2 = df['Filtered1']
        # y3 = df['Unfiltered2']
        # y4 = df['Filtered2']

        # Filter requirements.
        order = 6
        fs = 30.0  # sample rate, Hz
        cutoff = 0.2  # desired cutoff frequency of the filter, Hz

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
        # T = 5  # seconds
        # n = 120
        t = np.linspace(0, T, n, endpoint=False)

        # Filter the data, and plot both the original and filtered signals.
        y = butter_highpass_filter(datastream, cutoff, fs, order)
        # print(y)

        # if plot:
        #     plt.subplot(2, 1, 2)
        #     # plt.plot(t, data, 'b-', label='data')
        #     plt.plot(t, datastream, 'b-', label='data')
        #     plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
        #     plt.xlabel('Time [sec]')
        #     plt.grid()
        #     plt.legend()
        #     plt.subplots_adjust(hspace=0.35)
        #     plt.show()
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
    # participant = 3
    sensors = ['Accelerometer', 'Gravity', 'Orientation', 'Quaternion', 'RotationRate', 'UserAcceleration']

    print("Experiment 3")
    for participant in range(1, 4, 1):
        for sensor in sensors:

            path = "C://Users/giuse/Desktop/TotalOutputFiles/Experiment" + str(experiment) + "/participant" + str(participant) + "/phone" + sensor + "Exp" + str(experiment) + "Part" + str(participant) + ".csv"

            first = pd.read_csv(path, header=None)
            # print(first)

            first[1] = 5 - first[1]
            first = first.drop([5], axis=1).drop([0], axis=1).to_numpy()
            # print(first)
            # first[0] = 5 - first[0]
            # print(first)

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
                    datas = datas[rows - 120:]
                    # print(datas.shape)
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
            datas = datas[rows - 120:]
            # print(datas.shape)
            min = rows if rows < min else min
            # final_list.append(lol)
            final_list.append(datas)
            datas = []
            count += rows
            # print ("Final count: " + str(final_list.__len__()))
            # print(count)
            # print(min)
            transpose = np.transpose(final_list[0])
            # print(transpose)

            valuesSensorX = []
            valuesSensorY = []
            valuesSensorZ = []
            for row in final_list:
                current_row = np.transpose(row)
                if sensor == 'Orientation':
                    valuesSensorX.append(foo(1, current_row[1], True))
                    valuesSensorY.append(foo(1, current_row[2], True))
                    valuesSensorZ.append(foo(1, current_row[3], True))
                else:
                    valuesSensorX.append(foo(1, current_row[1], False))
                    valuesSensorY.append(foo(1, current_row[2], False))
                    valuesSensorZ.append(foo(1, current_row[3], False))

            # print(transpose[1])

            # print(foo(1, transpose[1]))
            newtranspose = []
            newtranspose.append(transpose[0])
            newtranspose.append(foo(1, transpose[1], False))
            newtranspose.append(foo(1, transpose[2], False))
            newtranspose.append(foo(1, transpose[3], False))

            np.savetxt("randomdata/watch" + sensor + "XExp"+ str(experiment) + "Part" + str(participant) + ".txt", valuesSensorX, delimiter=" ")
            np.savetxt("randomdata/watch" + sensor + "YExp"+ str(experiment) + "Part" + str(participant) + ".txt", valuesSensorY, delimiter=" ")
            np.savetxt("randomdata/watch" + sensor + "ZExp"+ str(experiment) + "Part" + str(participant) + ".txt", valuesSensorZ, delimiter=" ")

    experiment = 2
    print("Experiment 2")
    for participant in range(1, 4, 1):
        print('Participant: %d' % (participant))
        for sensor in sensors:
            print('Sensor: ' + sensor)
            valuesSensorX = []
            valuesSensorY = []
            valuesSensorZ = []
            for text_number in range (1, 12, 1):
                print('Txt number: %d' % (text_number))
                path = "C://Users/giuse/Desktop/TotalOutputFiles/Experiment" + str(experiment) + "/participant" + str(
                    participant) + "/watch" + sensor + "Exp" + str(experiment) + "Part" + str(
                    participant) + "_" + str(text_number).zfill(4) + ".csv"
                first = pd.read_csv(path, header=None)
                first = first.drop([5], axis=1).drop([0], axis=1).to_numpy()

                first = first[75:]
                rows, cols = first.shape
                first = first[:-(rows-120)]

                transpose = np.transpose(first)

                valuesSensorX.append(foo(1, transpose[1], False))
                valuesSensorY.append(foo(1, transpose[2], False))
                valuesSensorZ.append(foo(1, transpose[3], False))



            for text_number in range (1, 12, 1):
                if (text_number != 10):
                    path = "C://Users/giuse/Desktop/TotalOutputFiles/Experiment" + str(
                        experiment) + "/participant" + str(
                        participant) + "/watch" + sensor + "Exp" + str(experiment) + "Part" + str(
                        participant) + "_" + str(text_number).zfill(4) + ".csv"
                    first = pd.read_csv(path, header=None)
                    first = first.drop([5], axis=1).drop([0], axis=1).to_numpy()

                    first = first[105:]
                    rows, cols = first.shape
                    first = first[:-(rows-120)]

                    transpose = np.transpose(first)

                    valuesSensorX.append(foo(1, transpose[1], False))
                    valuesSensorY.append(foo(1, transpose[2], False))
                    valuesSensorZ.append(foo(1, transpose[3], False))

            for text_number in range (1, 12, 1):
                if (text_number != 10):
                    path = "C://Users/giuse/Desktop/TotalOutputFiles/Experiment" + str(
                        experiment) + "/participant" + str(
                        participant) + "/watch" + sensor + "Exp" + str(experiment) + "Part" + str(
                        participant) + "_" + str(text_number).zfill(4) + ".csv"
                    first = pd.read_csv(path, header=None)
                    first = first.drop([5], axis=1).drop([0], axis=1).to_numpy()

                    first = first[135:]
                    rows, cols = first.shape
                    first = first[:-(rows-120)]

                    transpose = np.transpose(first)

                    valuesSensorX.append(foo(1, transpose[1], False))
                    valuesSensorY.append(foo(1, transpose[2], False))
                    valuesSensorZ.append(foo(1, transpose[3], False))

            np.savetxt("randomdata/watch" + sensor + "XExp" + str(experiment) + "Part" + str(participant) + ".txt", valuesSensorX,
                       delimiter=" ")
            np.savetxt("randomdata/watch" + sensor + "YExp" + str(experiment) + "Part" + str(participant) + ".txt", valuesSensorY,
                       delimiter=" ")
            np.savetxt("randomdata/watch" + sensor + "ZExp" + str(experiment) + "Part" + str(participant) + ".txt", valuesSensorZ,
                       delimiter=" ")

    experiment = 2
    participant = 1
    sensor = 'Accelerometer'
    path = "randomdata/watch" + sensor + "XExp" + str(experiment) + "Part" + str(participant) + ".txt"
    # first = pd.read_csv(path1, header=None, delim_whitespace=True).to_numpy()
    path2 = "randomdata/watch" + sensor + "XExp" + str(experiment+1) + "Part" + str(participant+2) + ".txt"
    # second = pd.read_csv(path2, header=None, delim_whitespace=True).to_numpy()
    # concat = np.append(first, second)
    # print(concat)
    # np.savetxt("randomdata/test.txt", concat.tolist(), delimiter=" ")
    axis = ['X', 'Y', 'Z']
    num_lines = [29, 19, 19]

    for sensor in sensors:
        for axle in axis:
            filenames = []
            for exp in range(2):
                for participant in range(1, 4, 1):
                    filenames.append("randomdata/watch" + sensor + axle + "Exp" + str(experiment+exp) + "Part" + str(participant) + ".txt")
            order = [0, 5, 1, 4, 2, 3]

            with open("randomdata/formodel/watch" + sensor + axle + "_train.txt", "w") as outfile_train:
                with open("randomdata/formodel/watch" + sensor + axle + "_test.txt", "w") as outfile_test:
                    for position in order:
                        with open(filenames[position]) as infile:
                            contents = infile.readlines()
                            train = []
                            if (position == 0) or (position == 3):
                                train = contents[:25]
                                test = contents[25:29]
                            else:
                                train = contents[:17]
                                test = contents[17:19]
                            for line in train:
                                outfile_train.write(line)
                            for line in test:
                                outfile_test.write(line)

                                # outfile_train.write(contents[:19])
                                # outfile_test.write(contents[19:])
    # filenames = [path1, path2]
    # with open("randomdata/ftest.txt", "w") as outfile:
    #     for filename in filenames:
    #         with open(filename) as infile:
    #             contents = infile.read()
    #             outfile.write(contents)





# ----- ----- ----- ----- ----- -----
if __name__ == '__main__':
    main()