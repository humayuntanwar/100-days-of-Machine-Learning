import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell 

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
x = tf.placeholder('float32', [None, n_chunks, chunk_size])
Y = tf.placeholder('float32')

# functions neural network model


def recurrent_nueral_network_model(x):
    # weights now in one gaint tensor, biases added after weights , biases adds to that, input data x weights + biases
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x,[1,0,2])
    x = tf.reshape(x,[-1,chunk_size])
    x = tf.split(x,n_chunks,0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True) 
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1],layer['weights']) ,layer['biases'])
    return output




# input data > pass through model > output is one hot ARRAY
def train_neural_network(x):
    prediction = recurrent_nueral_network_model(x)
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
                epoch_X = epoch_X.reshape((batch_size, n_chunks,chunk_size))
                # run optimizer with cost
                _,c = sess.run([optimizer,cost],feed_dict = {x:epoch_X, Y:epoch_Y})
                epoch_loss+= c
            print('Epoch',epoch,'completed out of', hm_epochs, 'Loss',epoch_loss)
        #arg max will return index of max value inthese arrays
        # hopping both are same using equal
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
        #cast changes variable   and finding mean
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({x:mnist.test.images.reshape((-1,n_chunks,chunk_size)),Y:mnist.test.labels}))

train_neural_network(x)

