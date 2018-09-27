# for resizing images
import cv2
import numpy as np
#play with diretories
import os
from random import shuffle
#pretify view
from tqdm import tqdm
#train directory
TRIAN_DIR ="C:\\Users\\HumayunT\\Downloads\\Compressed\\all\\train\\train" 
#test directory
TEST_DIR ="C:\\Users\\HumayunT\\Downloads\\Compressed\\all\\test" 
IMG_SIZE = 50 #resizing img
LR =1e-3 #learning rate 0.001

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic') #to save the model

#process the data

# convert to np greyscale i2d image, makes it 2d array
# labels needs to one hot[cats, dogs] [1,0]
#takes img path
def label_img(img):
    #split by . and back 3 dog.93.png dog is neg3
    word_label = img.split('.')[-3]
    if word_label =='cat': return [1,0] # for cats
    elif word_label =='dog': return [0,1] # for dogs

#create trianing data
def create_trian_data():
    trianing_data = []
    #for img in there
    for img in tqdm(os.listdir(TRIAN_DIR)):
        label = label_img(img)
        #joining to get pull path
        path = os.path.join(TRIAN_DIR,img)
        #load in cv2, tuple 50x50
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
        #covert to nparray
        trianing_data.append([np.array(img),np.array(label)])
    #shuffle the data
    shuffle(trianing_data)
    #save the data
    np.save('train_data.npy',trianing_data)
    return trianing_data

#testing data fucntion
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),img_num])
    
    np.save('test_data.npy',testing_data)
    return testing_data

#train_data = create_trian_data()
#if you already have train data
train_data = np.load('train_data.npy')




##### CONVOLUTIONAL NEURAL NEtWORK

#imports
import tensorflow as tf
import tflearn
#convolution layers, max pool layers, dropouts , fully connected layers
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression


tf.reset_default_graph()
# defining input layer, shape and name
convnet = input_data(shape=[None,IMG_SIZE,IMG_SIZE,1],name='input')

#convolution pooling layer size = 32, windows=2, 
convnet = conv_2d(convnet,32,2,activation='relu')
convnet = max_pool_2d(convnet,2)

#convolution pooling layer 2  size = 64, windows=2, 
convnet = conv_2d(convnet,64,2,activation='relu')
convnet = max_pool_2d(convnet,2)

#convolution pooling layer size = 32, windows=2, 
convnet = conv_2d(convnet,32,2,activation='relu')
convnet = max_pool_2d(convnet,2)

#convolution pooling layer 2  size = 64, windows=2, 
convnet = conv_2d(convnet,64,2,activation='relu')
convnet = max_pool_2d(convnet,2)

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
convnet = fully_connected(convnet,2, activation='softmax')
#applying regression, all parameters selft explanitory
#calculating loss, regression layer
convnet = regression(convnet,optimizer='adam',learning_rate=LR, loss='categorical_crossentropy', name='targets')
#using the DNN models
model = tflearn.DNN(convnet,tensorboard_dir='log')


###DAY 30

#check if meta file exist
# load exsitnng model
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded')

#training data
train = train_data[:-500] #data and labels
#testing data
test = train_data[-500:]

#traing data
#feature set
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
#label
Y = [i[1] for i in train]

#testing data
#feature set
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
#label
test_y = [i[1] for i in test]

#training the model

model.fit({'input':X},{'targets':Y},n_epoch=2,
validation_set=({'input':test_x},{'targets':test_y}),
snapshot_step=500,show_metric=True,run_id=MODEL_NAME)


#launch tensorboard
#tensorboard --logdir=foo:C:\Users\HumayunT\Desktop\Pythonstart\MachineLearning\DogsvsCats\log
model.save(MODEL_NAME)
