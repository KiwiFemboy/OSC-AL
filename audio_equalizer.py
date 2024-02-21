import chunk
from posixpath import split
from tokenize import group
import pyaudio as pa
import numpy as np
import struct 
import matplotlib.pyplot as plt
import time

from pythonosc import udp_client

p = pa.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print ("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

# Get audio devices and choose one from this list


min_frequency = 20
max_frequency = 15000
CHUNK = 1024
FORMAT = pa.paInt16
CHANNELS = 1
RATE = 44100 # 44.1 HZ


osc_client = udp_client.SimpleUDPClient("127.0.0.1", 9000)


def calculate_channel_frequency(min_frequency, max_frequency, custom_channel_mapping,
                                custom_channel_frequencies):
    '''Calculate frequency values for each channel, taking into account custom settings.'''

    # How many channels do we need to calculate the frequency for
    if custom_channel_mapping != 0 and len(custom_channel_mapping) == 5:
        # print("Custom Channel Mapping is being used: %s", str(custom_channel_mapping))
        channel_length = max(custom_channel_mapping)
    else:
        # print("Normal Channel Mapping is being used.")
        channel_length = 5

    # print("Calculating frequencies for %d channels.", channel_length)
    octaves = (np.log(max_frequency / min_frequency)) / np.log(2)
    # print("octaves in selected frequency range ... %s", octaves)
    octaves_per_channel = octaves / channel_length
    frequency_limits = []
    frequency_store = []

    frequency_limits.append(min_frequency)
    if custom_channel_frequencies != 0 and (len(custom_channel_frequencies) >= channel_length + 1):
        # print("Custom channel frequencies are being used")
        frequency_limits = custom_channel_frequencies
    else:
        # print("Custom channel frequencies are not being used")
        for i in range(1, 5 + 1):
            frequency_limits.append(frequency_limits[-1]*2**octaves_per_channel)
            #frequency_limits.append(frequency_limits[-1]
            #                        * 10 ** (3 / (10 * (1 / octaves_per_channel))))
    for i in range(0, channel_length):
        frequency_store.append((frequency_limits[i], frequency_limits[i + 1]))
        # print("channel %d is %6.2f to %6.2f ", i, frequency_limits[i],
        #               frequency_limits[i + 1])

    # we have the frequencies now lets map them if custom mapping is defined
    if custom_channel_mapping != 0 and len(custom_channel_mapping) == 5:
        frequency_map = []
        for i in range(0, 5):
            mapped_channel = custom_channel_mapping[i] - 1
            mapped_frequency_set = frequency_store[mapped_channel]
            mapped_frequency_set_low = mapped_frequency_set[0]
            mapped_frequency_set_high = mapped_frequency_set[1]
            # print("mapped channel: " + str(mapped_channel) + " will hold LOW: "
            #               + str(mapped_frequency_set_low) + " HIGH: "
            #               + str(mapped_frequency_set_high))
            frequency_map.append(mapped_frequency_set)
        return frequency_map
    else:
        return frequency_store

def piff(val, sample_rate):
    '''Return the power array index corresponding to a particular frequency.'''
    return int(CHUNK * val / sample_rate)

# TODO(todd): Move FFT related code into separate file as a library
def calculate_levels(data, sample_rate, frequency_limits):
    '''Calculate frequency response for each channel
    
    Initial FFT code inspired from the code posted here:
    http://www.raspberrypi.org/phpBB3/viewtopic.php?t=35838&p=454041
    
    Optimizations from work by Scott Driscoll:
    http://www.instructables.com/id/Raspberry-Pi-Spectrum-Analyzer-with-RGB-LED-Strip-/
    '''

    # create a numpy array. This won't work with a mono file, stereo only.
    data_stereo = np.frombuffer(data, dtype=np.int16)
    # print(len(data), len(data)/4)
    data = np.empty(int(len(data) / 4))  # data has two channels and 2 bytes per channel
    data[:] = data_stereo[::2]  # pull out the even values, just using left channel

    # if you take an FFT of a chunk of audio, the edges will look like
    # super high frequency cutoffs. Applying a window tapers the edges
    # of each end of the chunk down to zero.
    window = np.hanning(len(data))
    data = data * window

    # Apply FFT - real data
    fourier = np.fft.rfft(data)

    # Remove last element in array to make it the same size as CHUNK_SIZE
    fourier = np.delete(fourier, len(fourier) - 1)

    # Calculate the power spectrum
    power = np.abs(fourier) ** 2

    matrix = [0 for i in range(4)]
    for i in range(4):
        # take the log10 of the resulting sum to approximate how human ears perceive sound levels
        matrix[i] = np.log10(np.sum(power[piff(frequency_limits[i][0], sample_rate)
                                          :piff(frequency_limits[i][1], sample_rate):1]))
        matrix[i] = matrix[i] - 8.0
        matrix[i] = matrix[i] / 4
        if matrix[i] < 0.05:
            matrix[i] = 0.0
        elif matrix[i] > 1.0:
            matrix[i] = 1.0

    return matrix


def main():
    p = pa.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        input_device_index=6,
        frames_per_buffer=CHUNK
    )

    while True:
        data = stream.read(CHUNK)
        frequency_limits = calculate_channel_frequency(min_frequency,
                                                   max_frequency,
                                                   0,
                                                   0)
        # dataInt = struct.unpack(str(2 * CHUNK) + 'B', data)
        grouped_data = calculate_levels(data, RATE, frequency_limits)
        
        print(grouped_data,end="\r")
        
        parameter = "/avatar/parameters/Bass"
        parameter1 = "/avatar/parameters/LowMid"
        parameter2 = "/avatar/parameters/HighMid"
        parameter3 = "/avatar/parameters/Treble"
        
        osc_client.send_message(parameter,grouped_data[0])
        osc_client.send_message(parameter1,grouped_data[1])
        osc_client.send_message(parameter2,grouped_data[2])
        osc_client.send_message(parameter3,grouped_data[3])

main()