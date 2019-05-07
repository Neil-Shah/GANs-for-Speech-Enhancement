import os, sys
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt 
import random
import scipy
import h5py
from scipy import io as sio
from scipy.io import savemat
from scipy.io import loadmat

###########################################  directory ############################################
mainfolder = "/home/Documents/WHSP2SPCH_MCEP"
submainfoder = mainfolder+"/vGAN/context/symmetric/11context"

directory_spectrum = submainfoder+'/test_spectrum'
if not os.path.exists(directory_spectrum):
    os.makedirs(directory_spectrum)

directory_model = submainfoder+'/model_pathGAN'
if not os.path.exists(directory_model):
    os.makedirs(directory_model)

loadtrainingpath = mainfolder+'/batches/Training_complementary_feats' # path to training features (.mat)
loadvalidationpath = mainfolder+'/batches/Validation_complementary_feats' # path to validation features (.mat)

data = sio.loadmat(loadtrainingpath + "/Batch_0.mat")
ip = data['Feat'] #11 context  1000X275
op = data['Clean_cent']

############################################### parameters ###############################################
n_input = ip.shape[1]
n_output = op.shape[1]

n_hidden1_gen = 512 
n_hidden2_gen = 512 
n_hidden3_gen = 512 

n_hidden1_dis = 512 
n_hidden2_dis = 512 
n_hidden3_dis = 512 
learning_rate = 0.001

training_epochs = 100
training_batches = 1242
validation_batches = 200
VALIDATION_RMSE = []

batch_size = ip.shape[0]

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)*0.01

###########################################################################################################
x = tf.placeholder(tf.float32, [None,n_input], name="noisybatch") # ip to G n/w
y_ = tf.placeholder(tf.float32, [None,n_output], name="centralcleanframe") # actual clean central spectrum, MSE loss at G n/w
MEAN = tf.placeholder(tf.float32, [1])
STD = tf.placeholder(tf.float32, [1])

weights = {
    'Gw1': tf.Variable(tf.random_normal([n_input, n_hidden1_gen])*0.0001, name="Gw1"),
    'Gw2': tf.Variable(tf.random_normal([n_hidden1_gen, n_hidden2_gen])*0.0001, name="Gw2"),
    'Gw3': tf.Variable(tf.random_normal([n_hidden2_gen, n_hidden3_gen])*0.0001, name="Gw3"),
    'Gwout': tf.Variable(tf.random_normal([n_hidden3_gen, op.shape[1]])*0.0001, name="Gwout"),

    'Dw1': tf.Variable(xavier_init([op.shape[1], n_hidden1_dis])*0.001, name="Dw1"),
    'Dw2': tf.Variable(xavier_init([n_hidden1_dis, n_hidden2_dis])*0.001, name="Dw2"),
    'Dw3': tf.Variable(xavier_init([n_hidden2_dis, n_hidden3_dis])*0.001, name="Dw3"),
    'Dwout': tf.Variable(xavier_init([n_hidden3_dis, 1])*0.001, name="Dwout")
}

biases = {
    'Gb1': tf.Variable(tf.random_normal(shape=[n_hidden1_gen])*0.01, name="Gb1"),
    'Gb2': tf.Variable(tf.random_normal(shape=[n_hidden2_gen])*0.01, name="Gb2"),
    'Gb3': tf.Variable(tf.random_normal(shape=[n_hidden3_gen])*0.01, name="Gb3"),
    'Gbout': tf.Variable(tf.random_normal(shape=[op.shape[1]])*0.01, name="Gbout"),

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
D_real, D_logit_real = discriminator((y_-MEAN)/STD)         # actual clean central ip to D
D_fake, D_logit_fake = discriminator((esti_spec-MEAN)/STD) # estimated clean central ip to D

# calculate the loss
D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_real), logits=D_logit_real)
D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_logit_fake), logits=D_logit_fake)

D_loss = tf.reduce_mean(D_loss_real) + tf.reduce_mean(D_loss_fake)
G_RMSE = 0.5*(tf.reduce_mean(tf.square(tf.subtract(y_, esti_spec))))
G_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit_fake), logits=D_logit_fake))
G_loss = G_gan + G_RMSE

D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list=theta_G)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

################################################### TRAINING ################################################################
k = 0
model_path = directory_model + "/model" + str(k) + ".ckpt"

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, model_path)

    for epoch in range(0,training_epochs):
        saver.restore(sess, model_path)

        Rand_files = np.random.permutation(training_batches)
        batch_index = 0
        for batch in Rand_files:

            data = sio.loadmat(loadtrainingpath + "/Batch_" + str(batch)+".mat")          
            
            batch_noisy_full = data['Feat']                      
            batch_noisy = batch_noisy_full[:,25:250]
            batch_cleancentral = data['Clean_cent'] 
            
            mean = np.array([np.mean(batch_cleancentral[:,:])])
            std = np.array([np.std(batch_cleancentral[:,:])])

            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={x: batch_noisy, y_: batch_cleancentral, MEAN: mean, STD: std})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={x: batch_noisy, y_: batch_cleancentral, MEAN: mean, STD: std})
            G_RMSE_curr = sess.run(G_RMSE, feed_dict={x: batch_noisy, y_: batch_cleancentral, MEAN: mean, STD: std})

            print("Epoch: "+str(epoch)+" Batch_index: "+str(batch_index)+" D_Cost= "+str(D_loss_curr)+ " G_cost= "+str(G_loss_curr)+ " G_rmse= "+str(G_RMSE_curr))
            batch_index = batch_index+1
            
        ################################################### validation ################################################################   
        RMSE = []
        for v_speech in range(0,validation_batches): 
            data = sio.loadmat(loadvalidationpath + "/Test_Batch_" + str(v_speech)+".mat") 

            batch_noisy_full = data['Feat']                      
            batch_noisy = batch_noisy_full[:,25:250]
            batch_cleancentral = data['Clean_cent'] 
            mean = np.array([np.mean(batch_cleancentral[:,:])])
            std = np.array([np.std(batch_cleancentral[:,:])])

            G_RMSE_curr = sess.run(G_RMSE, feed_dict={x: batch_noisy, y_: batch_cleancentral, MEAN: mean, STD: std})
            RMSE.append(G_RMSE_curr)

        print("After epoch "+str(epoch)+" Validation error is "+str(np.average(RMSE)))
        VALIDATION_RMSE.append(np.average(RMSE))
   
        k = k+1
        model_path = directory_model + "/model" + str(k) + ".ckpt"
        save_path = saver.save(sess, model_path)


scipy.io.savemat(submainfoder+"/"+str('Validation_errorganrmse.mat'),  mdict={'foo': VALIDATION_RMSE})
plt.figure(1)
plt.plot(VALIDATION_RMSE)
plt.savefig(submainfoder+'/validationerrorganrmse.png')
plt.show()