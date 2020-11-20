import numpy as np
import pandas as pd
import tensorflow as tf

def get_samples(num_gene, args):
    samples = []
    pos_file = args.input_positive
    neg_file = args.input_negative
    label = []
    disease = []
    micro = []
    with open(pos_file, "r") as f:
        for line in f:
            print(line)
            if line[0] ==' ':
                continue
            line_data = line.strip().split('\t')
            disease.append(line_data[0])
            micro.append(line_data[1])
            samples.append((line_data[0],line_data[1]))
            label.append([0, 1])

    with open(neg_file, "r") as f:
        for line in f:
            if line[0]==' ':
                continue
            line_data = line.strip().split('\t')
            samples.append((line_data[0],line_data[1]))
            disease.append(line_data[0])
            micro.append(line_data[1])
            label.append([1, 0])

    pheno_vector = pd.read_csv(args.input_pheno)
    miro_vector = pd.read_csv(args.input_miRNA)
    vocab_size = len(samples)

    W = np.zeros(shape=(vocab_size, num_gene), dtype='float32')
    W[0] = np.zeros(num_gene, dtype='float32')
    i = 0
    for sample in samples:  #398
        v1 = list(pheno_vector[sample[0]])
        v2 = list(miro_vector[sample[1]])
        v=[(v1[i]+v2[i])/2 for i in range(0,len(v1))]
        v1.extend(v2)
        v1.extend(v)
        W[i] = v1
        i = i + 1
    return W, label

#Generates a batch iterator for a dataset.
def batch_iter(data, batch_size, num_epochs, shuffle=True):

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

