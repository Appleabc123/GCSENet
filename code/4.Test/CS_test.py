import tensorflow as tf
import numpy as np
import os
import argparse
import data_helpers as dh
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score
from sklearn import metrics
from tensorflow.contrib import learn
import matplotlib.pyplot as plt
import sklearn.preprocessing as prep
from sklearn.metrics import average_precision_score
from tflearn.layers.conv import global_avg_pool





def parse_args():
    parser = argparse.ArgumentParser(description="Run CNN.")
    ## the input file
    ##disease-gene relationships and miRNA-gene relatiohships

    # parser.add_argument('--input_disease_gene', nargs='?', default='..\..\data\process_feature\output_file\disease_gene.csv',
    #                     help='Input disease_gene_relationship file')
    # parser.add_argument('--input_dmRNA_gene', nargs='?', default='..\..\data\process_feature\output_file\miRNA_gene.csv',
    #                     help='Input miRNA_gene_relationship file')
    parser.add_argument('--file', nargs='?', default='..\..\data\Test\\test.txt',
                        help='Input disease_gene_relationship file')
    parser.add_argument('--input_label',nargs = '?',default='..\..\data\CNN_SENet\label.csv',
                        help='sample label')
    parser.add_argument('--batch_size', nargs='?', default=64,
                        help = 'number of samples in one batch')                     
    parser.add_argument('--training_epochs', nargs='?', default=200,    #150
                        help= 'number of epochs in SGD')
    parser.add_argument('--display_step', nargs='?', default=10)
    parser.add_argument('--test_percentage', nargs='?', default=0.1,
                        help='percentage of test samples')
    parser.add_argument('--num_gene', nargs= '?', default = 1789,
                        help= 'number of genes related to disease and miRNA')
    parser.add_argument('--dev_percentage', nargs='?', default=0.1,
                        help='percentage of validation samples')
    parser.add_argument('--L2_norm', nargs='?', default=0.002,
                        help='percentage of validation samples')
    parser.add_argument('--keep_prob', nargs='?', default=0.5,
                        help='keep_prob when using dropout')
    parser.add_argument('--optimizer', nargs='?', default=tf.train.AdamOptimizer,
                        help='optimizer for learning weights')
    parser.add_argument('--learning_rate', nargs='?', default=1e-2,
                        help='learning rate for the SGD')
    return parser.parse_args()

def standard_scale(X_train):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    return X_train

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev= 0.1)
    weights = tf.Variable(initial)
    return weights

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding= "VALID")

def max_pool_2(x, W):
    return tf.nn.max_pool(x, ksize = W, strides= [1,10,1,1], padding= "VALID")

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def Relu(x):
    return tf.nn.relu(x)

def Fully_connected(x, units= 0, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)

def squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = Global_Average_Pooling(input_x)
        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = input_x * excitation
        return scale

def get_data(args):

    input_data, input_label = dh.get_samples(args)
    input_data = standard_scale(input_data)
    dev_sample_percentage = args.dev_percentage
    test_sample_percentage = args.test_percentage
    x = np.array(input_data)
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(input_label)))
    input_data = [x[i] for i in shuffle_indices]
    input_label = [input_label[i] for i in shuffle_indices]

    dev_sample_index = -2 * int(dev_sample_percentage * float(len(input_label)))
    test_sample_index = -1 * int(test_sample_percentage * float(len(input_label)))

    x_train, x_dev, test_data = input_data[:dev_sample_index],input_data[dev_sample_index:test_sample_index],\
                                input_data[test_sample_index:]
    y_train, y_dev, test_label = input_label[:dev_sample_index], input_label[dev_sample_index:test_sample_index], \
                                 input_label[test_sample_index:]

    return test_data,test_label


def deepnn(x, keep_prob, args):
    with tf.name_scope('reshape'):
        x = tf.reshape(x, [-1, 2048, 1, 1])

    with tf.name_scope('conv_pool'):
        filter_shape = [4, 1, 1, 4]
        W_conv = weight_variable(filter_shape)
        b_conv = bias_variable([4])
        h_conv = tf.nn.relu(conv2d(x, W_conv) + b_conv)
        h_pool = tf.nn.max_pool(h_conv, ksize = [1, 4, 1, 1], strides= [1,4,1,1], padding= "VALID")
        h_pool= squeeze_excitation_layer(h_pool, 4 , 4, layer_name='senet')
        regula = tf.contrib.layers.l2_regularizer(args.L2_norm)
        h_input1 = tf.reshape(h_pool,[-1, 511 * 4])
        W_fc1 = weight_variable([511 * 4, 50])
        b_fc1 = bias_variable([50])
        h_input2 = tf.nn.relu(tf.matmul(h_input1, W_fc1) + b_fc1)
        h_keep = tf.nn.dropout(h_input2, keep_prob)
        W_fc2 = weight_variable([50, 2])
        b_fc2 = bias_variable([2])
        h_output = tf.matmul(h_keep, W_fc2) + b_fc2
        regularizer = regula(W_fc1) + regula(W_fc2)
        return h_output, regularizer

def main(args):
    with tf.device('/cpu:0'):
        test_data = dh.get_samples(args.num_gene,args)
        with tf.name_scope('input'):
            input_data = tf.placeholder(tf.float32, [None, 2048],name='input_data')
            input_label = tf.placeholder(tf.float32, [None, 2],name='input_label')
        keep_prob = tf.placeholder(tf.float32)
        y_conv, losses = deepnn(input_data, keep_prob, args)
        y_res = tf.nn.softmax(y_conv)
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=input_label)
        cross_entropy = tf.reduce_mean(cross_entropy)
        los = cross_entropy + losses
        with tf.name_scope('optimizer'):
            optimizer = args.optimizer
            learning_rate = args.learning_rate
            train_step = optimizer(learning_rate).minimize(los)
        with tf.name_scope('accuracy'):
            predictions = tf.argmax(y_conv, 1)
            correct_predictions = tf.equal(predictions, tf.argmax(input_label, 1))
            correct_predictions = tf.cast(correct_predictions, tf.float32)
            accuracy = tf.reduce_mean(correct_predictions,name='accuracy')

        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('./model')
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("111111")
                sess.run(tf.global_variables_initializer())

            y_predict = sess.run(y_res, feed_dict={input_data: test_data, keep_prob :1.0})[:, 1]


if __name__ == '__main__':
    args = parse_args()
    main(args)


    












