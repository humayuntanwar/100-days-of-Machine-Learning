#imports
import tflearn
#convolution layers, max pool layers, dropouts , fully connected layers
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

#assigning the datasets to our variables
X,Y, test_x, test_y = mnist.load_data(one_hot=True)

#reshape x and text_x to  28x28 size
X = X.reshape([-1,28,28,1])
test_x = test_x.reshape([-1,28,28,1])

# defining input layer, shape and name
convnet = input_data(shape=[None,28,28,1],name='input')

#convolution pooling layer size = 32, windows=2, 
convnet = conv_2d(convnet,32,2,activation='relu')
convnet = max_pool_2d(convnet,2)

#convolution pooling layer 2  size = 64, windows=2, 
convnet = conv_2d(convnet,64,2,activation='relu')
convnet = max_pool_2d(convnet,2)

#convolution pooling  fully connected layer   size =1024 , windows=2, 
convnet = fully_connected(convnet,1024, activation='relu')
# dropout to convet, dropoutrate = 0.8
convnet = dropout(convnet,0.8)

# now for activation as softmax
convnet = fully_connected(convnet,10, activation='softmax')
#applying regression, all parameters selft explanitory
#calculating loss, regression layer
convnet = regression(convnet,optimizer='atom',learning_rate=0.01, loss='categorical_crossentropy', name='targets')
#using the DNN models
model = tflearn.DNN(convnet)

#training models with the given parameters accroding to tflearn function

model.fit({'input':X},{'targets':Y},n_epoch=2,
validation_set=({'input':test_x},{'targets':test_y}),
snapshot_step=500,show_metric=True,run_id='mnist')

#saving model
model.save('tflearncn.model')

#loading model
model.load('quicktest.model')

#Testing our model
import numpy as np
print( np.round(model.predict([test_x[1]])[0]) )
print(test_y[1])