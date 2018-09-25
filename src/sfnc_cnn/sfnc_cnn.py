# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import matplotlib.pyplot as plt

# functions
# generate a data batch for training/testing the model
# Return a total of `num` random samples and labels.
def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    newdata = np.reshape(data[idx], (num,-1))
    return newdata, labels[idx]

def plot_fnc(fnc_mat, save=''):
    plt.imshow(fnc_mat, cmap=plt.get_cmap('jet'))
    plt.colorbar()
    plt.clim(-1, 1)
    plt.xticks(np.arange(fnc_mat.shape[0])+.5)
    plt.yticks(np.arange(fnc_mat.shape[0])+.5)
    plt.tick_params(axis='both', which='both', top='off', right='off', bottom='off', left='off', labelbottom='off', labelleft='off')
    plt.minorticks_on()
    plt.grid(color='w', which='major', axis='both', linewidth=.4)
    if save:
        plt.savefig(save, dpi=300)

# constants
outpath = '../../results/zfu_data_analysis/sfnc/'

# image parameters
numIC = 47
num_subject = 314
width = numIC
height = numIC
channels = 1
n_inputs = height * width * channels

# tensorflow parameters
n_epochs = 100
batch_size = 32
use_gpu = 1

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 2


# load data
u_data = np.ndfromtxt('../../results/meta_314_subjects.csv', delimiter=',', 
                      dtype=[(str,80),(str,80),float,float])
labels = u_data['f3']-1

# input data numpy array
sfnc_gigica = loadmat(outpath+'/sfnc_gigica.mat')
sfnc_gigica = np.transpose(sfnc_gigica['sfnc_gigica']);
images = []
for j in range(num_subject):
    a = np.zeros((numIC,numIC))
    a[np.triu_indices(numIC,1)] = sfnc_gigica[j,:]
    a = a + np.flipud(np.rot90(a))
    images.append(a)

images = np.array(images)

accuracy_test = []

for j in range(10):
    # split into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)
    X_valid, X_train = X_train[:batch_size], X_train[batch_size:]
    y_valid, y_train = y_train[:batch_size], y_train[batch_size:]
    
    # enable/disable GPU
    config = tf.ConfigProto(
            device_count = {'GPU': use_gpu}
        )
    
    best_loss = np.infty
    epochs_without_progress = 0
    max_epochs_without_progress = 10

    # define the graph
    tf.reset_default_graph()

    with tf.name_scope("inputs"):
        X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
        X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
        y = tf.placeholder(tf.int32, name="y")
    
    conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                             strides=conv1_stride, padding=conv1_pad,
                             activation=tf.nn.relu, name="conv1")
    conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                             strides=conv2_stride, padding=conv2_pad,
                             activation=tf.nn.relu, name="conv2")
    
    with tf.name_scope("pool3"):
        pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        pool3_flat = tf.reshape(pool3, shape=[-1, int(np.prod(pool3.shape[1:4]))])
    
    with tf.name_scope("fc1"):
        fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")
    
    with tf.name_scope("output"):
        logits = tf.layers.dense(fc1, n_outputs, name="output")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")
    
    with tf.name_scope("train"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)
    
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    # experiment
    with tf.Session(config=config) as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(len(X_train) // batch_size):
                X_batch, y_batch = next_batch(batch_size, X_train, y_train)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            
            X_batch, y_batch = next_batch(batch_size, X_valid, y_valid)
            accuracy_val, loss_val = sess.run([accuracy, loss], feed_dict={X: X_batch, y: y_batch})
            print(epoch, "Validation accuracy:", accuracy_val, "loss:", loss_val)
            
#            X_batch, y_batch = next_batch(len(X_test), X_test, y_test)
#            accuracy_test = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            
            if loss_val < best_loss:
#                save_path = saver.save(sess, outpath+"/cnn/cnn_model"+str(j))
                best_loss = loss_val
            else:
                epochs_without_progress += 1
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break
        
        X_batch, y_batch = next_batch(len(X_test), X_test, y_test)
        accuracy_test.append( accuracy.eval(feed_dict={X: X_batch, y: y_batch}) )

print(accuracy_test)




