import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist data
# one hot, one component will be hot , dictate something
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
hm_epochs = 2

# num of classes
n_classes = 10
# batches of 100 features at a time
batch_size = 128

chunk_size = 28
n_chunks = 28
rnn_size = 128

# height x width 
x = tf.placeholder('float32', [None, 784])
Y = tf.placeholder('float32')

#takes one pixels at a time usings strides
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def maxpool2d(x):
    #                       size of window     movement of window
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# functions neural network model
def convolutional_nueral_network_model(x):

    weights= {'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
            'W_conv2': tf.Variable(tf.random_normal([5,5,32,64])),
            'W_fc': tf.Variable(tf.random_normal([7*7*64,1024])),
            'out': tf.Variable(tf.random_normal([1024,n_classes])),}

    biases= {'b_conv1': tf.Variable(tf.random_normal([32])),
            'b_conv2': tf.Variable(tf.random_normal([64])),
            'b_fc': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([n_classes])),}
    
    x = tf.reshape(x,shape=[-1,28,28,1])
    
    conv1 = conv2d(x,weights['W_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = conv2d(conv1,weights['W_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1,7*7*64])
    #relu = rectified linear
    fc = tf.nn.relu(tf.matmul(fc,weights['W_fc'])+ biases['b_fc'])

    output = tf.add(tf.matmul(fc,weights['out']),biases['out'])
    return output




# input data > pass through model > output is one hot ARRAY
def train_neural_network(x):
    prediction = convolutional_nueral_network_model(x)
    #calculate the difference of prediction we got to known label we have
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y) )    
    # using adamoptimizer synonymous with gradient descent, minimize cost  
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #how many epochs we want , do a now num, if slow comp
    # cycles feed forward +backprop
    
#began and run session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #calcualate loss as we go
        for epoch in range(hm_epochs):
            epoch_loss = 0
            #total num of samples / batch size to see how many cycles we need to run
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # chuncks through dataset
                epoch_X,epoch_Y = mnist.train.next_batch(batch_size)
                # run optimizer with cost
                _,c = sess.run([optimizer,cost],feed_dict = {x:epoch_X, Y:epoch_Y})
                epoch_loss+= c
            print('Epoch',epoch,'completed out of', hm_epochs, 'Loss',epoch_loss)
        #arg max will return index of max value inthese arrays
        # hopping both are same using equal
        correct = tf.equal(tf.arg_max(prediction,1),tf.argmax(Y,1))
        #cast changes variable   and finding mean
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({x:mnist.test.images,Y:mnist.test.labels}))

train_neural_network(x)

