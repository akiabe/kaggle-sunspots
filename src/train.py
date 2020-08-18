import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import config
import utils

if __name__ == "__main__":
    # load dataset
    time, series = utils.sunspotsDataset(config.TRAINING_FILE)

    # split train and valid set
    split_time = 3000
    train_time, train_x, valid_time, valid_x = utils.split_dataset(time, series, split_time)

    # Set the constans that pass to the windowed_dataset function
    window_size = 60
    batch_size = 32
    shuffle_buffer_size = 1000

    # set the dataset
    train_set = utils.windowed_dataset(
        train_x,
        window_size,
        batch_size,
        shuffle_buffer_size
    )

    # build the dense model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(20, input_shape=[window_size], activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    # set optimizer
    optimizer = tf.keras.optimizers.SGD(lr=1e-7, momentum=0.9)

    # compile model
    model.compile(
        loss="mse",
        optimizer=optimizer
    )

    # train the model
    history = model.fit(
        train_set,
        epochs=100,
        verbose=1
    )

    print("TRAINING COMPLETE !")

    forecast = []

    for time in range(len(series) - window_size):
        forecast.append(model.predict(series[time:time+window_size][np.newaxis]))

    forecast = forecast[split_time-window_size:]
    results = np.array(forecast)[:,0,0]

    plt.figure(figsize=(10, 6))

    utils.plot_series(valid_time, valid_x)
    utils.plot_series(valid_time, results)



