import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def sunspotsDataset(PATH):
    # create empty list
    time_step = []
    sunspots = []

    with open(PATH) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # read the columns title and end up throwing it away
        next(reader)

        # loop through the reader and append data into list
        for row in reader:
            sunspots.append(float(row[2]))
            time_step.append(int(row[0]))

        # convert list into numpy array
        series = np.array(sunspots)
        time = np.array(time_step)

    return series, time

def split_dataset(time, series, split_time):
    train_time = time[:split_time]
    train_x = series[:split_time]
    valid_time = time[split_time:]
    valid_x = series[split_time:]

    return train_time, train_x, valid_time, valid_x

def windowed_dataset(series, window_size, batch_size, shuffle_buffer_size):
    # create dataset from series and pass the series to it
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # window the dataset
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)

    # mapping a dataset element to a (window+1) batch size dataset
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))

    # shuffle the dataset at the shuffle_buffer and split into features and labels
    dataset = dataset.shuffle(shuffle_buffer_size).map(lambda window: (window[:-1], window[-1]))

    # batch the select batch size
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset


