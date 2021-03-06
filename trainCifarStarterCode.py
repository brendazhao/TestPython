from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp
import os

# --------------------------------------------------
# setup

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

    #return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

    #return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    #return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #return h_max


def create_summaries(var, name):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary('y/' + name, var)
    return None



result_dir = './results/' # directory where the results from the training are saved

ntrain =  1000 # per class
ntest =    100 # per class
nclass =    10 # number of classes
imsize =    28
nchannels = 1
batchsize = 100

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = 'CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

sess = tf.InteractiveSession()

tf_data = tf.placeholder("float", shape=[None,imsize,imsize,nchannels]) #tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_labels = tf.placeholder("float", shape=[None,nclass]) #tf variable for labels

# --------------------------------------------------
# model
#create your model
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(tf_data, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, nclass])
b_fc2 = bias_variable([nclass])

forward = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)






W1_a = W_conv1  # [5, 5, 1, 32]
W1pad = tf.zeros([5, 5, 1, 1])  # [5, 5, 1, 4]  - four zero kernels for padding
# We have a 6 by 6 grid of kernepl visualizations. yet we only have 32 filters
# Therefore, we concatenate 4 empty filters
W1_b = tf.concat(3, [W1_a, W1pad, W1pad, W1pad, W1pad])  # [5, 5, 1, 36]
W1_c = tf.split(3, 36, W1_b)  # 36 x [5, 5, 1, 1]
W1_row0 = tf.concat(0, W1_c[0:6])  # [30, 5, 1, 1]
W1_row1 = tf.concat(0, W1_c[6:12])  # [30, 5, 1, 1]
W1_row2 = tf.concat(0, W1_c[12:18])  # [30, 5, 1, 1]
W1_row3 = tf.concat(0, W1_c[18:24])  # [30, 5, 1, 1]
W1_row4 = tf.concat(0, W1_c[24:30])  # [30, 5, 1, 1]
W1_row5 = tf.concat(0, W1_c[30:36])  # [30, 5, 1, 1]
W1_d = tf.concat(1, [W1_row0, W1_row1, W1_row2, W1_row3, W1_row4, W1_row5])  # [30, 30, 1, 1]
W1_e = tf.reshape(W1_d, [1, 30, 30, 1])
Wtag = tf.placeholder(tf.string, None)
image_summary_t = tf.image_summary("Visualize_kernels", W1_e)



# --------------------------------------------------
# loss
#set up the loss, optimization, evaluation, and accuracy
#cross_entropy = -tf.reduce_sum(tf_labels*tf.log(forward))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_labels * tf.log(forward), reduction_indices=[1]))
#optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#optimizer = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(cross_entropy)
#optimizer = tf.train.MomentumOptimizer(1e-3, 0.9).minimize(cross_entropy)

evaluation = tf.equal(tf.argmax(forward,1), tf.argmax(tf_labels,1))
accuracy = tf.reduce_mean(tf.cast(evaluation, "float"))



# Add a scalar summary for the snapshot loss.
tf.scalar_summary(cross_entropy.op.name, cross_entropy)
# Build the summary operation based on the TF collection of Summaries.
create_summaries(W_conv1, "Weight_conv1")
create_summaries(h_conv1, "h_conv1")
create_summaries(h_conv2, "h_conv2")
summary_op = tf.merge_all_summaries()

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# Instantiate a SummaryWriter to output summaries and the Graph.
summary_writer = tf.train.SummaryWriter(result_dir, sess.graph)


# --------------------------------------------------
# optimization

sess.run(tf.initialize_all_variables())
batch_xs = np.zeros((batchsize,imsize,imsize,nchannels)) #setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros((batchsize,nclass)) #setup as [batchsize, the how many classes]

nsamples = nclass * ntrain
max_step = 2500
for i in range(max_step): # try a small iteration size once it works then continue
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
        batch_ys[j,:] = LTrain[perm[j],:]

    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
        #print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))
        #calculate train accuracy and print it

    # save the checkpoints every 10 iterations
    if i % 100 == 0 or i == max_step:
	# Update the events file which is used to monitor the training (in this case,
	# only the training loss is monitored)
        summary_str = sess.run(summary_op, feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()
        checkpoint_file = os.path.join(result_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=i)


    optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5}) # dropout only during training

# --------------------------------------------------
# test

print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))


sess.close()
