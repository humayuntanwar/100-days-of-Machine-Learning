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
initial_games = 100

def some_random_games_first():
    # Each of these is its own game
    for episode in range(5):
        # this is each frame, up to 200...but we wont make it that far.
        env.reset()
        for t in range(goal_steps):
            #render env , remove if you wana run fast
            env.render()
            #generate random action
            action = env.action_space.sample()
            #obs=array of data from game actual pixel data(pole pos, car pos) , rew=1 0r 1, done = game over, info=any info
            observation,reward,done, info = env.step(action)
            if done:
                break

some_random_games_first()


