import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
'''
input data >weight it > hidden lay 1 (activation function) 
> weights > hidden layer 2(activation layer) > weights > poutput layer

compare output to intended output >cost function(cross entropy)

optimization function(optimizer) > minimize cost(adamoptimizer .. SGD, AdaGrad)

back propagation

feed forward + backdrop = epoch

'''
#mnist data
# one hot, one component will be hot , dictate something
mnist = input_data.read_data_sets("/tmp/data",one_hot=True)

#nodes for hidden layer s
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#num of classes
n_classes = 10
#batches of 100 features at a time
batch_size = 100
# height x width 
x= tf.placeholder('float32',[None,784])
Y= tf.placeholder('float32')

# functions neural network model

def nueral_network_model(data):
    #weights now in one gaint tensor, biases added after weights , biases adds to that, input data x weights + biases
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2 ,n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

    # model
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.multiply(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.multiply(l2, hidden_3_layer['weights']) , hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.multiply(l3, output_layer['weights']) + output_layer['biases']    
    return output


#Day 19

# specify how to run data through the model in session

# input data > pass through model > output is one hot ARRAY
def train_neural_network(x):
    prediction = nueral_network_model(x)
    #calculate the difference of prediction we got to known label we have
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y) )    
    # using adamoptimizer synonymous with gradient descent, minimize cost  
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #how many epochs we want , do a now num, if slow comp
    # cycles feed forward +backprop
    hm_epochs = 2
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
        correct = tf.equal(tf.arg_max(prediction,1),tf.arg_max(Y,1))
        #cast changes variable   and finding mean
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({x:mnist.test.images,Y:mnist.test.labels}))

train_neural_network(x)

