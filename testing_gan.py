import os, sys
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt 
import random
import scipy
from scipy import io as sio
from scipy.io import savemat
import h5py
import pickle

###########################################  directory ############################################
mainfolder = "/home/Documents/"
submainfoder = mainfolder + "/GAN"

directory_spectrum = submainfoder+'/test_spectrum'
directory_model = submainfoder+'/model_pathGAN'

loadtestingpath =mainfolder+'/Testing_complementary_feats' # path to extracted test features

# decide which model to load
model_path = directory_model + "/model" + str(100) + ".ckpt" # Best model from the validation data

############################################### parameters ###############################################
# no.of noisy features
number = 108

data = sio.loadmat(loadtestingpath+'/Test_Batch_0.mat')
noisy_contxt_full = data['Feat']  #11-context                    
noisy_contxt = noisy_contxt_full[:,25:250] #9-context

n_hidden1_gen = 512 
n_hidden2_gen = 512 
n_hidden3_gen = 512 

n_hidden1_dis = 512 
n_hidden2_dis = 512 
n_hidden3_dis = 512 

n_input = noisy_contxt.shape[1]
n_output = 1

learning_rate = 0.001

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)*0.01

###########################################################################################################
x = tf.placeholder(tf.float32, [None,n_input], name="noisybatch") # ip to G n/w
y_ = tf.placeholder(tf.float32, [None,n_output], name="centralcleanframe") # actual clean central spectrum, MSE loss at G n/w

weights = {
    'Gw1': tf.Variable(tf.random_normal([n_input, n_hidden1_gen])*0.0001, name="Gw1"),
    'Gw2': tf.Variable(tf.random_normal([n_hidden1_gen, n_hidden2_gen])*0.0001, name="Gw2"),
    'Gw3': tf.Variable(tf.random_normal([n_hidden2_gen, n_hidden3_gen])*0.0001, name="Gw3"),
    'Gwout': tf.Variable(tf.random_normal([n_hidden3_gen, n_output])*0.0001, name="Gwout"),

    'Dw1': tf.Variable(xavier_init([n_output, n_hidden1_dis])*0.001, name="Dw1"),
    'Dw2': tf.Variable(xavier_init([n_hidden1_dis, n_hidden2_dis])*0.001, name="Dw2"),
    'Dw3': tf.Variable(xavier_init([n_hidden2_dis, n_hidden3_dis])*0.001, name="Dw3"),
    'Dwout': tf.Variable(xavier_init([n_hidden3_dis, 1])*0.001, name="Dwout")
}

biases = {
    'Gb1': tf.Variable(tf.random_normal(shape=[n_hidden1_gen])*0.01, name="Gb1"),
    'Gb2': tf.Variable(tf.random_normal(shape=[n_hidden2_gen])*0.01, name="Gb2"),
    'Gb3': tf.Variable(tf.random_normal(shape=[n_hidden3_gen])*0.01, name="Gb3"),
    'Gbout': tf.Variable(tf.random_normal(shape=[n_output])*0.01, name="Gbout"),

    'Db1': tf.Variable(tf.zeros(shape=[n_hidden1_dis]), name="Db1"),
    'Db2': tf.Variable(tf.zeros(shape=[n_hidden2_dis]), name="Db2"),
    'Db3': tf.Variable(tf.zeros(shape=[n_hidden3_dis]), name="Db3"),
    'Dbout': tf.Variable(tf.zeros(shape=[1]), name="Dbout")
}

theta_G = [weights['Gw1'], weights['Gw2'], weights['Gw3'], weights['Gwout'], 
            biases['Gb1'], biases['Gb2'], biases['Gb3'], biases['Gbout']]
theta_D = [weights['Dw1'], weights['Dw2'], weights['Dw3'], weights['Dwout'], 
            biases['Db1'], biases['Db2'], biases['Db3'], biases['Dbout']]

def generator(x, weights, biases):    
    layer_1 = tf.nn.relu(tf.add(tf.matmul((x), weights['Gw1']), biases['Gb1'])) # 500x512
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['Gw2']), biases['Gb2'])) # 500x512
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['Gw3']), biases['Gb3'])) # 500x512
    esti_spec = tf.add(tf.matmul(layer_3, weights['Gwout']), biases['Gbout'], name='mask') # 500x64
    return esti_spec

def discriminator(X):
    layer_1 = tf.nn.tanh(tf.matmul(X, weights['Dw1']) + biases['Db1'])
    layer_2 = tf.nn.tanh(tf.matmul(layer_1, weights['Dw2']) + biases['Db2'])
    layer_3 = tf.nn.tanh(tf.matmul(layer_2, weights['Dw3']) + biases['Db3'])
    logit = tf.matmul(layer_3, weights['Dwout']) + biases['Dbout']
    prob = tf.nn.sigmoid(logit, name="D_prob")
    return prob, logit

# construct the model
esti_spec = generator(x, weights, biases) # noisy context ip to G

init = tf.global_variables_initializer()
saver = tf.train.Saver()

#########################################  Main  ########################################
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, model_path)
    for i in range(0, number):

        # get the data
        data = sio.loadmat(loadtestingpath+'/Test_Batch_'+ str(i)+ '.mat')

        noisy_contxt_full = data['Feat']                 
        noisy_contxt = noisy_contxt_full[:,25:250]

        # find batch size of incoming data
        batch_size = noisy_contxt.shape[0]
            
        # obtain the predicted mask
        pred_spectrum = sess.run(esti_spec, feed_dict={x: noisy_contxt})  

        file = directory_spectrum+ '/File_'+ str(i)+ '.mat'
        scipy.io.savemat(file,  mdict={'PRED_SPEC': pred_spectrum})
        print("file"+str(i))