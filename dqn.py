from keras.models import Sequential
from keras.layers import Dense, Flatten, Permute, Conv2D
from keras.optimizers import RMSprop
from collections import deque
import numpy as np
import random
import gym
from datetime import datetime, timedelta


def calculate_eps(start_eps, end_eps, current_step, num_steps):
    # epsilon will take linear steps toward the end epsilon value until the final step where eps=end_eps
    # the suggested values from the paper are start_eps=1.0, end_eps=0.1, and num_steps=1000000
    if current_step > num_steps:
        return end_eps
    v = -(start_eps - end_eps)/num_steps
    return min(v*current_step+max(start_eps, end_eps), 1)


start_time = datetime.now()
timecounter = datetime.now()


# set up the environment
env = gym.make('VideoPinball-v0')


# dataset used for experience replay
D = deque(maxlen=1000000)
gamma = 0.99
minibatch_size = 32


# recreate the CNN used in the paper
# use an input shape of 4 frames of 84x84 pixels
window_size = (84,84)
state_frames = (4,)
input_shape = state_frames+window_size


model = Sequential()

model.add(Permute((2, 3, 1), input_shape=input_shape))  # Fixes image_dim_ordering issue
model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu"))

model.add(Flatten())
model.add(Dense(512, activation="relu"))

model.add(Dense(env.action_space.n, activation="linear"))

model.compile(loss='mse', optimizer=RMSprop(lr=0.00025), metrics=['mae'])


print("============= Model Built =============")
model.summary()




