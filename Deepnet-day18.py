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
X= tf.placeholder('float',[None,784])
Y= tf.placeholder('float')

# functions neural network model

def nueral_network_model(data):
    #weights now in one gaint tensor, biases added after weights , biases adds to that, input data x weights + biases
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'biases':tf.Variable(tf.random_normal(n_nodes_hl2))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2 ,n_nodes_hl3])),'biases':tf.Variable(tf.random_normal(n_nodes_hl3))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),'biases':tf.Variable(tf.random_normal(n_classes))}

    # model
    l1 = tf.add(tf.multiply(data, hidden_1_layer['weights']) + hidden_1_layer['biases'])
    


