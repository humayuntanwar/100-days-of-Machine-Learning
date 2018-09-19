import numpy as np
import tensorflow as tf
from create_sentimentfeature_set_day20 import create_feature_sets_and_labels
train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

#nodes for hidden layer s
n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

#num of classes
n_classes = 2
#batches of 100 features at a time
batch_size = 100
# height x width 
x= tf.placeholder('float',[None,len(train_x[0])])
y= tf.placeholder('float')

# functions neural network model

def nueral_network_model(data):
    #weights now in one gaint tensor, biases added after weights , biases adds to that, input data x weights + biases
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2 ,n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

    # model
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']) , output_layer['biases'])
    return output


#Day 19

# specify how to run data through the model in session

# input data > pass through model > output is one hot ARRAY
def train_neural_network(x):
    prediction = nueral_network_model(x)
    #calculate the difference of prediction we got to known label we have
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2( logits=prediction, labels=y) )
    # using adamoptimizer synonymous with gradient descent, minimize cost  
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #how many epochs we want , do a now num, if slow comp
    # cycles feed forward +backprop
    hm_epochs = 10
#began and run session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #calcualate loss as we go
        for epoch in range(hm_epochs):
            epoch_loss = 0
            #total num of samples / batch size to see how many cycles we need to run
            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _,c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch',epoch+1,'completed out of', hm_epochs, 'Loss',epoch_loss)
        #arg max will return index of max value inthese arrays
        # hopping both are same using equal
        correct = tf.equal(tf.arg_max(prediction,1),tf.argmax(y,1))
        #cast changes variable   and finding mean
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({x:test_x,y:test_y}))

train_neural_network(x)

