import numpy as np
import pandas as pd
import tensorflow as tf


def get_samples(args):
    train_data = pd.read_csv(args.input_disease_miRNA).T
    train_data = np.array(train_data).tolist()
    print('done reading data')
    train_label = pd.read_csv(args.input_label, header=None)
    print("done reading label")
    train_label = np.array(train_label).tolist()
    return train_data, train_label
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
