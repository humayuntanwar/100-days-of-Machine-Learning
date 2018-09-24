#open ai gym
import gym
#got the game env and agent will move randmomly
import random
# numpy stuff
import numpy as np
import tflearn
#input layer, dropout 20%, typical fully connected layer
from tflearn.layers.core import input_data,dropout,fully_connected
#regression for final layer
from tflearn.layers.estimator import regression
# to show how well random did
from statistics import mean, median
from collections import Counter

#learning rate
LR = 1e-3
#env game name
env = gym.make('CartPole-v0')
# start env
env.reset()
#every frame we go , we get a score on every frame , if pole balance +1 to score
goal_steps = 500
#min score
score_requirement = 50
#num of times games run
initial_games = 10000

def some_random_games_first():
    # Each of these is its own game
    for episode in range(5):
        # this is each frame, up to 200...but we wont make it that far.
        env.reset()
        for t in range(goal_steps):
            #render env , remove if you wana run fast
            #env.render()
            #generate random action
            action = env.action_space.sample()
            #obs=array of data from game actual pixel data(pole pos, car pos) , rew=1 0r 1, done = game over, info=any info
            observation,reward,done, info = env.step(action)
            if done:
                break

#defining Intial Population of Data
# we can generate training samples
def initial_population():
    #empty list, the actual data we will trian on , observation on moves mades, appened only if score above 50
    training_data = []
    #empty list
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        score = 0
        #store all movements in gm
        game_memory = []
        prev_observation = []

        for _ in range(goal_steps):

            #genrate 0s and 1s
            action = random.randrange(0,2)
            observation,reward,done, info = env.step(action)

            # if we had prev_obs then append to game memory
            if len(prev_observation) > 0:
                game_memory.append([prev_observation,action])
            
            prev_observation = observation
            score += reward

            if done:
                break

        if score >= score_requirement:
            # if acceptable
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] ==1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                training_data.append([data[0],output])
        #game over
        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)

    print('Average accepted scores:',mean(accepted_scores))
    print('median accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

#### DAY 27

#some_random_games_first()

#initial_population()
#create our nueral network model
# input_size ,sperate model from triaing and testing
def nueral_network_model(input_size):
    #input layer
    network = input_data(shape=[None,input_size,1],name='input')
    #full connected layer
    network = fully_connected(network,128,activation='relu')
    network = dropout(network,0.8)
    
    network = fully_connected(network,256,activation='relu')
    network = dropout(network,0.8)

    network = fully_connected(network,512,activation='relu')
    network = dropout(network,0.8)

    network = fully_connected(network,256,activation='relu')
    network = dropout(network,0.8)

    network = fully_connected(network,128,activation='relu')
    network = dropout(network,0.8)

    #output layer
    network = fully_connected(network,2,activation='softmax')
    network = regression(network,optimizer='adam',learning_rate=LR,loss='categorical_crossentropy',name='targets')

    model = tflearn.DNN(network,tensorboard_dir='log')

    return model

#train model
def train_model(training_data, model=False):
    #observations
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0]),1)
    y = [i[1] for i in training_data]    
    
    if not model:
        model = nueral_network_model(input_size =len(X[0]))
    
    model.fit({'input':X},{'targets':y},n_epoch=3,snapshot_step=500,show_metric=True,run_id='openaistuff')

    return model

training_data = initial_population()
model = train_model(training_data)
