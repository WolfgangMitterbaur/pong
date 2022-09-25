# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 20:39:44 2022

@author: Wolfgang Mitterbaur

file name solutiondqncnn.py

DQN deep Q-learning artificial network to learn playing pong
"""

# all imports
import numpy as np
import gym
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import time
from collections import deque
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt

class Experiences(object):
    '''
    class to store all experiences
    one experience is a single step in a episode
    '''
    def __init__(self, max_len):
        '''
        class constructur
        max_len: maximal length of the double-ended queue
        '''
        self.max_len = max_len                    # maximal length of double-ended queue
        self.frames = deque(maxlen=max_len)       # queue of all frames
        self.actions = deque(maxlen=max_len)      # queue of all actions
        self.rewards = deque(maxlen=max_len)      # queue of all rewards
        self.done = deque(maxlen=max_len)         # queue of all done-flags

    def add_experience(self, frame, action, reward, done):
        '''
        public method to add one experience
        frame: next frame to add
        action: next action to add
        reward: next reward to add
        done: next done-flag to add
        '''
        self.frames.append(frame)
        self.actions.append(action)
        self.rewards.append(reward)
        self.done.append(done)

class New_Experiences(object):
    '''
    class to store new experiences of a single episode
    one experience is a single step in a episode
    '''
    def __init__(self):
        '''
        class constructur
        '''
        self.frames = []                    # list of all frames af a single episode
        self.actions = []                   # list of all actions af a single episode
        self.rewards = []                   # list of all rewards af a single episode
        self.done = []                      # list of all done-flags of a singe episode

    def add_experience(self, frame, action, reward, done):
        '''
        public method to add one experience
        frame: next frame to add
        action: next action to add
        reward: next reward to add
        done: next done-flag to add
        '''
        self.frames.append(frame)
        self.actions.append(action)
        self.rewards.append(reward)
        self.done.append(done)

    def clear(self):
        '''
        public method to delete all experiences of the episode
        '''
        self.frames = []
        self.actions = []
        self.rewards = []
        self.done = []

class AnalyseFrame(object):
    '''
    class to analyse a single frame
    '''

    def __init__(self):
        '''
        class constructur
        '''
        self.lst_ball_posx = deque(maxlen=2)        # queue of ball positions x
        self.lst_ball_posy = deque(maxlen=2)        # queue of ball positions y
        self.lst_my_paddle_posx = deque(maxlen=2)   # queue of my paddle positions x
        self.lst_comp_paddle_posx = deque(maxlen=2) # queue of computer paddle positions x
        self.hit_ball = False                       # flag when the paddle hit the ball
        self.ball_posx = 0                          # the position x of the ball
        self.ball_posy = 0                          # the position y of the ball
        self.my_paddle_pos_min = 0                  # the minimum position x of my paddle
        self.my_paddle_pos_max = 0                  # the maximum position y of my paddle
        self.comp_paddle_pos_min = 0                # the minimum position x of the computer paddle
        self.comp_paddle_pos_max = 0                # the maximum position x of the computer paddle
        self.ball_movex = 0                         # the movement of the ball in x
        self.ball_movey = 0                         # the movement of the ball in y
        self.my_paddle_movex = 0                    # the movement of my paddle in x
        self.com_paddle_movex = 0                   # the movement of the computer´s paddle in x

    def clear(self):
        '''
        public method to delete all stored positions
        '''
        self.lst_ball_posx.clear()
        self.lst_ball_posy.clear()
        self.lst_my_paddle_posx.clear()
        self.lst_comp_paddle_posx.clear()

    def calc_positions(self, frame):
        '''
        public method to calculate positions of the ball, my paddle and computer paddle
        frame: the frame to be analyzed
        '''
        self.ball_posx = 0                          # the position x of the ball
        self.ball_posy = 0                          # the position y of the ball
        self.my_paddle_posx_min = 0                 # the minimum position x of my paddle
        self.my_paddle_posx_max = 0                 # the maximum position y of my paddle
        self.comp_paddle_posx_min = 0               # the minimum position x of the computer paddle
        self.comp_paddle_posx_max = 0               # the maximum position x of the computer paddle

        # loop over all lines (x-cooridnates)
        for i in range(0, 79):

            # search for the ball between y = 10 and y = 70
            for j in range(10, 70):
                if frame[i, j] > 100:
                    self.ball_posx = i
                    self.ball_posy = j

            # search for my paddle between y = 70 and y = 71
            for j in range(70, 71):
                if frame[i, j] > 100:
                    if self.my_paddle_posx_min == 0:
                        self.my_paddle_posx_min = i
                    else:
                        self.my_paddle_posx_max = i

            # search for the computer paddle between y = 8 and y = 9
            for j in range(8, 9):
                if frame[i, j] > 100:
                    if self.comp_paddle_posx_min == 0:
                        self.comp_paddle_posx_min = i
                    else:
                        self.comp_paddle_posx_max = i

        my_paddle_posx_a = (self.my_paddle_posx_min +
                            self.my_paddle_posx_max)/2
        comp_paddle_posx_a = (self.comp_paddle_posx_min +
                              self.comp_paddle_posx_max)/2

        # add the positions to the queue
        self.lst_ball_posx.append(self.ball_posx)
        self.lst_ball_posy.append(self.ball_posy)
        self.lst_my_paddle_posx.append(my_paddle_posx_a)
        self.lst_comp_paddle_posx.append(comp_paddle_posx_a)

        return self.ball_posx, self.ball_posy, my_paddle_posx_a, comp_paddle_posx_a

    def calc_movements(self):
        '''
        public method to calculate movement of the components
        '''
        
        # calculate the movement using the last two positions
        self.ball_movex = 0
        if len(self.lst_ball_posx) == 2:
            ball_velox = self.lst_ball_posx[1] - self.lst_ball_posx[0]

            if ball_velox > 0:
                self.ball_movex = 2
            elif ball_velox < 0:
                self.ball_movex = 1
            else:
                self.ball_movex = 0

        # calculate the movement using the last two positions
        self.ball_movey = 0
        if len(self.lst_ball_posy) == 2:
            ball_veloy = self.lst_ball_posy[1] - self.lst_ball_posy[0]

            if ball_veloy > 0:
                self.ball_movey = 2
            elif ball_veloy < 0:
                self.ball_movey = 1
            else:
                self.ball_movey = 0

        # calculate the movement using the last two positions
        self.my_paddle_movex = 0
        if len(self.lst_my_paddle_posx) == 2:
            my_paddle_velox = self.lst_my_paddle_posx[1] - \
                self.lst_my_paddle_posx[0]

            if my_paddle_velox > 0:
                self.my_paddle_movex = 2
            elif my_paddle_velox < 0:
                self.my_paddle_movex = 1
            else:
                self.my_paddle_movex = 0

        # calculate the movement using the last two positions
        self.comp_paddle_movex = 0
        if len(self.lst_comp_paddle_posx) == 2:
            comp_paddle_velox = self.lst_comp_paddle_posx[1] - \
                self.lst_comp_paddle_posx[0]

            if comp_paddle_velox > 0:
                self.comp_paddle_movex = 2
            elif comp_paddle_velox < 0:
                self.comp_paddle_movex = 1
            else:
                self.comp_paddle_movex = 0

        return self.ball_movex, self.ball_movey, self.my_paddle_movex, self.comp_paddle_movex

    def ball_hit_reward(self):
        '''
        public method to calculate a reward when the paddle hits the ball
        '''
        # save the event, the ball has been it to generate only one event        
        hit_ball_reward = 0
        if self.ball_posy < 60:
            self.hit_ball = False

        # check if my paddle hit the ball
        if self.ball_posy <= 69 and self.ball_posy >= 68 and not self.hit_ball:

            if (self.my_paddle_pos_min - 1 <= self.ball_posx and self.my_paddle_pos_max + 1 >= self.ball_posx)\
                    or (self.my_paddle_pos_min - 2 <= self.ball_posx and self.my_paddle_pos_min + 1 >= self.ball_posx and self.my_paddle_movex == 1)\
                    or (self.my_paddle_pos_max + 2 >= self.ball_posx and self.my_paddle_pos_max - 1 <= self.ball_posx and self.my_paddle_movex == 2):

                self.hit_ball = True
                hit_ball_reward = 0.3       # a small positive reward if the my paddle hit the ball

        return hit_ball_reward

class GameEnvironment(object):
    '''
    class of the game environment
    '''

    def __init__(self, name):
        '''
        class constructur
        name: name of the gym game
        '''
        self.analyse = AnalyseFrame()
        self.env = self._make_env(name)
        
    def _make_env(self, name):
        '''
        private method to delete all positions
        name: the name of the game
        '''
        #env = gym.make(name, render_mode='human')  # render mode to visualize
        env = gym.make(name)                        # without rendering for learning
        return env

    def take_step(self, agent, score):
        '''
        public method to play the next step
        agent: the agent, which controls the game
        score: the current score of this episode
        '''
        # update timesteps of the current episode
        agent.total_timesteps += 1

        # perform a step on the environment with the action of the last experience
        next_frame, next_reward, next_done, info = self.env.step(agent.new_experiences.actions[-1])

        # resize the frame of the next frame
        next_frame = downgrade_frame(next_frame)

        # analyse the frame
        self.analyse.calc_positions(next_frame)
        self.analyse.calc_movements()

        # check if my paddle hit the ball
        hit_ball_reward = self.analyse.ball_hit_reward()

        # create a new state
        state = [agent.new_experiences.frames[-3], agent.new_experiences.frames[-2], agent.new_experiences.frames[-1], next_frame]
        
        state = np.moveaxis(state, 0, 2) / 255 # keras's format [batch_size, rows, columns, channels]
        state = np.expand_dims(state, 0)

        # get next action, using next state
        next_action = agent.get_action(state)

        # if game is over, return the score and True
        if next_done:
            agent.new_experiences.add_experience(
                next_frame, next_action, next_reward + hit_ball_reward, next_done)
            return (score + next_reward), True

        # add the experience to the nex experience buffer
        agent.new_experiences.add_experience(
            next_frame, next_action, next_reward + hit_ball_reward, next_done)

        # if enough data available in the experience buffer start to learn
        if len(agent.experiences.frames) > agent.starting_mem_len:
            agent.learn()

        return (score + next_reward), False

    def take_test_step(self, agent, score):
        '''
        public method to play the next step
        agent: the agent, which controls the game
        score: the current score of this episode
        '''
        # update timesteps and save weights
        agent.total_timesteps += 1

        # get next action, using the last state: [1-] fetches the last state

        # perform a step on the environment
        next_frame, next_reward, next_done, info = self.env.step(
            agent.experiences.actions[-1])

        # resize the frame of the next step
        next_frame = downgrade_frame(next_frame)

        # create a sate
        state = [agent.experiences.frames[-3], agent.experiences.frames[-2],
                 agent.experiences.frames[-1], next_frame]
        
        state = np.moveaxis(state, 0, 2) / 255  # keras's format [batch_size,rows,columns,channels]
        state = np.expand_dims(state, 0)

        # get next action, using next state
        next_action = agent.get_test_action(state)

        # if game is over, return the score
        if next_done:
            agent.experiences.add_experience(
                next_frame, next_action, next_reward, next_done)
            return (score + next_reward), True

        # add the next experience to experience queue
        agent.experiences.add_experience(
            next_frame, next_action, next_reward, next_done)

        return (score + next_reward), False

    def close_game(self):
        '''
        public method to close the game
        '''
        self.env.close()

    def reset_game(self):
        '''
        public method to reset the game
        '''
        self.env.reset()

class Agent(object):
    '''
    class of agent to learn playing pong using a CNN
    '''

    def __init__(self, test_game):
        '''
        class constructur
        '''
        self.test_game = test_game                      # a test game is active
        self.experiences = Experiences(500000)          # experiences of all episodes
        self.new_experiences = New_Experiences()        # experiences of a single episode
        self.possible_actions = [0, 2, 3]               # all possible actions 0=stay, 2=up, 3=down
        self.epsilon = 1                                # start value of epsilon for exploration
        self.epsilon_delta = .9/100000                  # delta reduction of epsilon
        self.epsilon_min = 0.05                         # minimum value of epsilon
        self.gamma = 0.95                               # gamma the target update
        self.learn_rate = 0.00025                       # learn rate for the adam optimizer
        self.model = self._build_model()                # CNN learn model
        self.model_target = clone_model(self.model)     # target learn model
        self.total_timesteps = 0                        # total time steps
        self.starting_mem_len = 3000                    # start value to learn
        self.steps = 0                                  # learn steps
        self.batch_size = 32                            # batch size to fit neural network
        self.ddqn = False                               # double dqn network
        
    def _build_model(self):
        '''
        private method to create the models
        '''
        if self.test_game:
            model = tf.keras.models.load_model('11092022.h5') # this is used to load a finished model
        else:
        
            model = Sequential()
            model.add(Input((80, 80, 4)))
            model.add(Conv2D(filters = 32, kernel_size=(8, 8), strides = 4, data_format="channels_last",
                      activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale = 2)))
            model.add(Conv2D(filters = 64, kernel_size=(4, 4), strides = 2, data_format="channels_last",
                      activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale = 2)))
            model.add(Conv2D(filters = 64, kernel_size=(3, 3), strides = 1, data_format="channels_last",
                      activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale = 2)))
            model.add(Flatten())
            model.add(Dense(512, activation='relu',
                      kernel_initializer=tf.keras.initializers.VarianceScaling(scale = 2)))
            model.add(Dense(len(self.possible_actions), activation='linear'))
            optimizer = Adam(self.learn_rate)
            model.compile(optimizer, loss=tf.keras.losses.Huber(),
                          jit_compile = True)
            model.summary()
        
        return model
    
    def _index_valid(self, index):
        '''
        private method to detect if the episode is finished
        index: the current index of the experience queue
        '''
        if self.experiences.done[index-3] or self.experiences.done[index-2] or self.experiences.done[index-1] or self.experiences.done[index]:
            return False
        else:
            return True
        
    def get_action(self, state):
        '''
        public method to fetch the next action
        state: the state to predict an action
        '''
        # exploration: epsilon greedy procedure
        if np.random.rand() < self.epsilon:
            return random.sample(self.possible_actions, 1)[0]

        # expedation: calcualate the best action
        a_index = np.argmax(self.model.predict(state, verbose=0))
        return self.possible_actions[a_index]

    def get_test_action(self, state):
        '''
        public method to fetch the next action
        state: the state to predict an action
        '''

        a_index = np.argmax(self.model.predict(state, verbose=0))
        return self.possible_actions[a_index]

    def learn(self):
        '''
        public method to learn the CNN model
        '''
        # create a minibatch
        states = []                 # the states in the minibatch
        next_states = []            # the next states in the minibatch
        actions = []                # the action take at states in the minibatch
        next_rewards = []           # the reward of states in the minibatch
        next_done = []              # the done-flag in the minibatch

        while len(states) <  self.batch_size:

            # take a random index of the experience queue
            index = np.random.randint(4, len(self.experiences.frames) - 1)

            if self._index_valid(index):

                state = [self.experiences.frames[index-3], self.experiences.frames[index-2],
                         self.experiences.frames[index-1], self.experiences.frames[index]]
                state = np.moveaxis(state, 0, 2) / 255
                next_state = [self.experiences.frames[index-2], self.experiences.frames[index-1],
                              self.experiences.frames[index], self.experiences.frames[index+1]]
                
                next_state = np.moveaxis(next_state, 0, 2) / 255 # keras's format of [batch_size,rows,columns,channels]

                states.append(state)
                next_states.append(next_state)
                                
                actions.append(self.experiences.actions[index])
                next_rewards.append(self.experiences.rewards[index + 1])
                next_done.append(self.experiences.done[index + 1])

        # calculate the output of the model and the target model to estimate the error function
        target = self.model.predict(np.array(states), verbose = 0)
        
        if self.ddqn:
            target_next = self.model.predict(np.array(next_states), verbose = 0) #ddqn
        
        # predict Q-values for next state using the target model
        target_value = self.model_target.predict(np.array(next_states), verbose = 0)


        # define the target to train the model
        for i in range(len(states)):
            action = self.possible_actions.index(actions[i])
            
            if self.ddqn:
                a = np.argmax(target_next[i])
                target[i][action] = next_rewards[i] + (not next_done[i]) * self.gamma * target_value[i][a] #DDQN
            else:
                target[i][action] = next_rewards[i] + (not next_done[i]) * self.gamma * max(target_value[i]) #DQN
            
        # train the model using the states and output
        self.model.fit(np.array(states), target, batch_size = self.batch_size, epochs = 1, verbose = 0)
        
        # reduce the epsilon for the exploration/expedetion trade-off
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_delta
        self.steps += 1

        # update the target model every 10.000 learning steps
        if self.steps % 10000 == 0:
            self.model_target.set_weights(self.model.get_weights())

def modify_reward(reward):
    '''
    public method to create a reward function by rewarding
    reward: the list of rewards
    '''
    gamma = 0.9                                         # discount factor
    additive = 0                                  # running additive value
    modified_reward = np.zeros_like(reward)             # create discounted reward
    for i in reversed(range(0, len(reward))):

        if reward[i] != 0:
            additive = 0

        additive = additive * gamma + reward[i]

        if additive < 0.05 and additive > -0.05:
            additive = 0

        modified_reward[i] = additive

    return modified_reward

def downgrade_frame(frame):
    '''
    public method to downgrade the frame
    frame: the  the list of rewards
    '''
    frame = frame[35:195:2, ::2, :]                 # cut the image - remove the frame
    frame = np.average(frame, axis=2)
    frame = np.array(frame, dtype=np.uint8)         # remove the color

    return frame

def play_game(game, agent):
    '''
    public method to play a game
    game: the game environment
    agent: the agent with the learning model
    '''
    # reset the environment
    game.reset_game()
    # create a start frame and a empty experience at the beginning
    starting_frame = downgrade_frame(game.env.step(0)[0])
    for i in range(3):
        agent.new_experiences.add_experience(starting_frame, 0, 0, False)

    # play a new game
    done = False
    score = 0
    while True:
        score, done = game.take_step(agent, score)
        if done:
            break
    return score

def play_test_game(game, agent):
    '''
    public method to play a test game without learning
    game: the game environment
    agent: the agent with the learning model
    '''
    # reset the environment
    game.reset_game()
    # create a start frame and a empty experience at the beginning
    starting_frame = downgrade_frame(game.env.step(0)[0])
    for i in range(3):
        agent.experiences.add_experience(starting_frame, 0, 0, False)

    # play a new game
    done = False
    score = 0
    while True:
        score, done = game.take_test_step(agent, score)
        if done:
            break
    return score

def main():
    '''
    main function to start the program
    generates the game environment and the agent
    start all episodes
    '''
    name = 'PongDeterministic-v4'               # name of the gym game
    #pltname = 'Pong-v0'
    #name = 'Pong-v4'
    
    test_game = False                           # False = Train, True = load a existing agent
    no_episodes = 100                           # mumber of episodes
    scores = deque(maxlen = no_episodes)        # save the last 100 scores
    max_score = -21                             # intialize the saved maximum score
      
    game = GameEnvironment(name)
    agent = Agent(test_game)
             
    # play episodes
    for i in range(no_episodes):
    
        timesteps = agent.total_timesteps
        timee = time.time()
        
        if test_game:
            score = play_test_game(game, agent)
            print('score: ' + str(score))
        else:
            score = play_game(game, agent) 
            
            modified_reward = modify_reward(agent.new_experiences.rewards)
        
            for j in range(len(agent.new_experiences.frames)):
        
                next_frame = agent.new_experiences.frames[j]
                next_action = agent.new_experiences.actions[j]
                next_reward = modified_reward[j]
                next_done = agent.new_experiences.done[j]
        
                agent.experiences.add_experience(
                    next_frame, next_action, next_reward, next_done)
        
            # delete the exeriences from the last episode
            agent.new_experiences.clear()
            
            print('\nepisode: ' + str(i))
            print('steps: ' + str(agent.total_timesteps - timesteps))
            print('duration: ' + str(time.time() - timee))
            print('score: ' + str(score))
            print('max. score: ' + str(max_score))
            print('epsilon: ' + str(agent.epsilon))
            
        scores.append(score)
        if score > max_score:
            max_score = score
    
    plt.plot(scores)    
    ipythondisplay.clear_output(wait = True)
    game.close_game()

'''
main
'''
if __name__ == '__main__':
    main()
