# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 20:39:44 2022

@author: Wolfgang Mitterbaur

file name solutiondqn.py

DQN deep Q-learning artificial network to learn playing pong
"""

# all imports
import numpy as np
import gym
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import random
import time
import tensorflow as tf
from collections import deque
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt

class Experiences():
    '''
    class to store all experiences
    one experience is a single step in a episode
    '''
    def __init__(self,max_len):
        '''
        class constructur
        max_len: maximal length of the double-ended queue
        '''
        self.max_len = max_len                                      # maximal length of double-ended queue
        self.ball_posx = deque(maxlen = max_len)                    # position of the ball in x direction
        self.ball_posy = deque(maxlen = max_len)                    # position of the ball in y direction
        self.ball_velox = deque(maxlen = max_len)                   # the velocity of the ball in x direction
        self.ball_veloy = deque(maxlen = max_len)                   # the velocity of the ball in y direction
        self.my_paddle_posx = deque(maxlen = max_len)               # the position of the paddle in x direction
        self.my_paddle_velox = deque(maxlen = max_len)              # the position of the paddle in x direction
        self.comp_paddle_posx = deque(maxlen = max_len)             # the position of the paddle in x direction
        self.action = deque(maxlen = max_len)                       # the taken action
        self.reward = deque(maxlen = max_len)                       # the received reward
        self.done = deque(maxlen = max_len)                         # done-flag when episode stopped
        
    def add_experience(self, ball_posx, ball_posy, ball_velox, ball_veloy, my_paddle_posx, my_paddle_velox, comp_paddle_posx, action, reward, done):
        '''
        public method to add one experience
        ball_posx: the position of the ball in x direction
        ball_posy: the position of the ball in y direction
        ball_velox: the velocity of the ball in x direction
        ball_veloy: the velocity of the ball in y direction
        my_paddle_posx: the position of my paddle in x direction
        my_paddle_velox: the velocity of my paddle in x direction
        comp_paddle_posx: the positon of the computer´s padlle in x direction
        action: the taken acton
        reward: the received rewars
        done: done-flag when the eposisode ends        
        '''
        self.ball_posx.append(ball_posx)
        self.ball_posy.append(ball_posy)
        self.ball_velox.append(ball_velox)
        self.ball_veloy.append(ball_veloy)
        self.my_paddle_posx.append(my_paddle_posx)
        self.my_paddle_velox.append(my_paddle_velox)
        self.comp_paddle_posx.append(comp_paddle_posx)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        
class New_Experiences():
    '''
    class to store new experiences of a single episode
    one experience is a single step in a episode
    '''
    def __init__(self):
        '''
       class constructur
       '''
        self.ball_posx = []                             # list of all ball positons in x direction
        self.ball_posy =  []                            # list of all ball positions in y direction
        self.ball_velox = []                            # list of all ball velocities in x direction
        self.ball_veloy =  []                           # list of all ball velocities in y diretions
        self.my_paddle_posx =  []                       # list of all my paddle positions in x directions
        self.my_paddle_velox =  []                      # list of all my paddle velocity in x directions
        self.comp_paddle_posx =  []                     # list of al computer´s paddle positons in x direcion
        self.action = []                                # list of all actions
        self.reward = []                                # list of all rewards
        self.done = []                                  # list of all done-flags
        
    def add_experience(self, ball_posx, ball_posy, ball_velox, ball_veloy, my_paddle_posx, my_paddle_velox, comp_paddle_posx, action, reward, done):
        '''
        public method to add one experience
        ball_posx: the positon of the ball in x direction
        ball_posx: the position of the ball in y direction
        ball_velox: the velocity of the ball in x direction
        my_paddle_posx: the positon of my paddle in x direction
        my_paddle_velox: the velocity of my paddle in x direction
        comp_paddle_posx: the positon of the computer´s paddle in x direction
        action: the taken action
        reward: the received rewards
        done: the done-flag
        '''
        self.ball_posx.append(ball_posx)
        self.ball_posy.append(ball_posy)
        self.ball_velox.append(ball_velox)
        self.ball_veloy.append(ball_veloy)
        self.my_paddle_posx.append(my_paddle_posx)
        self.my_paddle_velox.append(my_paddle_velox)
        self.comp_paddle_posx.append(comp_paddle_posx)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        
    def clear(self):
        '''
        public method to delete all experiences of the episode
       '''
        self.ball_posx = []                                
        self.ball_posy = []                                
        self.ball_velox = []                                
        self.ball_veloy = []                                
        self.my_paddle_posx = []                            
        self.my_paddle_velox = []                           
        self.comp_paddle_posx = []                          
        self.action = []                                    
        self.reward = []                                    
        self.done = []                                      

class AnalyseFrame(object):
    '''
    class to analyse a single frame
    '''
    def __init__(self):
        '''
        class constructur
        ''' 
        self.lst_ball_posx = deque(maxlen = 2)              # queue of ball positions x
        self.lst_ball_posy = deque(maxlen = 2)              # queue of ball positions y
        self.lst_my_paddle_posx = deque(maxlen = 2)         # queue of  my paddle position x
        self.lst_comp_paddle_posx = deque(maxlen = 2)       # queue of computer´s paddle position x
        self.ball_posx = 0                                  # the position of the ball in x direction
        self.ball_posy = 0                                  # the position of the ball in y direction
        self.ball_velox = 0                                 # the speed of the ball in x direction
        self.ball_veloy = 0                                 # the speed of the ball in y direction
        self.my_paddle_pos_min = 0                          # the minimum position x of my paddle
        self.my_paddle_pos_max = 0                          # the maximum position y of my paddle
        self.comp_paddle_pos_min = 0                        # the minimum position x of the computer paddle
        self.comp_paddle_pos_max = 0                        # the maximum position x of the computer paddle
        self.my_paddle_velox = 0                            # the speed of my paddle in x
        self.com_paddle_velox = 0                           # the speed of the computer´s paddle in x
        self.hit_ball = False                               # flag when the paddle hit the ball
        
    def clear(self):
        '''
        public method to delete all stored positions
        ''' 
        self.lst_ball_posx.clear();
        self.lst_ball_posy.clear();
        self.lst_my_paddle_posx.clear();
        self.lst_comp_paddle_posx.clear();
    
    def calc_positions(self, frame):
        '''
        public method to calculate positions of the ball, my paddle and the computer paddle
        frame: the frame to be analyzed
        ''' 
        self.ball_posx = 0                              # the position x of the ball
        self.ball_posy = 0                              # the position y of the ball
        self.my_paddle_posx_min = 0                     # the minimum position x of my paddle
        self.my_paddle_posx_max = 0                     # the maximum position y of my paddle
        self.comp_paddle_posx_min = 0                   # the minimum position x of the computer´s paddle
        self.comp_paddle_posx_max = 0                   # the maximum position x of the computer´s paddle
            
        # loop over all lines (x-cooridnates)
        for i in range (0,79):
            
            # search for the ball between y = 10 and y = 70
            for j in range (10,70):
                if frame[i,j] > 100:
                    self.ball_posx = i;
                    self.ball_posy = j;
            
            # search for my paddle between y = 70 and y = 71
            for j in range (70,71):
               if frame[i,j] > 100:
                   if self.my_paddle_posx_min == 0:
                       self.my_paddle_posx_min = i;
                   else:
                       self.my_paddle_posx_max = i;
           
            # search for the computer paddle between y = 8 and y = 9
            for j in range (8,9):
                if frame[i,j] > 100:
                    if self.comp_paddle_posx_min == 0:
                        self.comp_paddle_posx_min = i;
                    else:
                        self.comp_paddle_posx_max = i;
        
        
        my_paddle_posx_a = (self.my_paddle_posx_min + self.my_paddle_posx_max)/2
        comp_paddle_posx_a = (self.comp_paddle_posx_min + self.comp_paddle_posx_max)/2
    
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
        # calculate the velocity using the last two positions
        self.ball_velox = 0                           
        if len(self.lst_ball_posx) == 2:
            self.ball_velox = self.lst_ball_posx[1] - self.lst_ball_posx[0]
            
        if self.ball_velox > 5 or self.ball_velox < -5:
            self.ball_velox = 0                           
                       
        # calculate the movement using the last two positions
        self.ball_veloy = 0                           
        if len(self.lst_ball_posy) == 2:
            self.ball_veloy = self.lst_ball_posy[1] - self.lst_ball_posy[0]
        if self.ball_veloy > 5 or self.ball_veloy < -5:
            self.ball_veloy = 0                           
        
        # calculate the movement using the last two positions
        self.my_paddle_velox = 0 
        if len(self.lst_my_paddle_posx) == 2:
            self.my_paddle_velox = self.lst_my_paddle_posx[1] - self.lst_my_paddle_posx[0]
        if self.my_paddle_velox > 5 or self.my_paddle_velox < -5:
            self.my_paddle_velox = 0 
            
        # calculate the movement using the last two positions
        self.comp_paddle_velox = 0 
        if len(self.lst_comp_paddle_posx) == 2:
            self.comp_paddle_velox = self.lst_comp_paddle_posx[1] - self.lst_comp_paddle_posx[0]
        if self.comp_paddle_velox > 5 or self.comp_paddle_velox < -5:
            self.comp_paddle_velox = 0 
      
                
        return self.ball_velox, self.ball_veloy, self.my_paddle_velox, self.comp_paddle_velox
    
    def ball_hit_reward(self):
        '''
        public method to calculate a reward when the paddle hits the ball
        '''                
        # save the event, the ball has been it to generate only one event
        hit_ball_reward = 0
 
        if self.ball_posy < 60:
            self.hit_ball = False
    
        if self.ball_posy <= 69 and self.ball_posy >= 68 and not self.hit_ball:
            
            if (self.my_paddle_pos_min - 1 <= self.ball_posx and self.my_paddle_pos_max + 1 >= self.ball_posx)\
                or (self.my_paddle_pos_min - 2 <= self.ball_posx and self.my_paddle_pos_min + 1 >= self.ball_posx and self.my_paddle_velox < 0)\
                or (self.my_paddle_pos_max + 2 >= self.ball_posx and self.my_paddle_pos_max - 1 <= self.ball_posx and self.my_paddle_velox > 0):
                
                self.hit_ball = True            
                hit_ball_reward = 0.3       # a small positive reward if the my paddle hit the ball              

        return hit_ball_reward
    
class GameEnvironment():
    '''
    class of the game environment
    one experience is a single step in a episode
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
        env = gym.make(name)
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
        next_frame, next_reward, next_done, info = self.env.step(agent.new_experiences.action[-1])
            
        # resize the frame of the next frame
        next_frame = self._downgrade_frame(next_frame)
        
        # analyse the frame        
        ball_posx, ball_posy, my_paddle_posx, comp_paddle_posx = self.analyse.calc_positions(next_frame)
        ball_velox, ball_veloy, my_paddle_velox, comp_paddle_velox = self.analyse.calc_movements()
         
        # check if my paddle hit the ball
        hit_ball_reward = self.analyse.ball_hit_reward()
        
        # normalize
        ball_posx = ball_posx / 80
        ball_posy = ball_posy / 80
        my_paddle_posx = my_paddle_posx / 80
        comp_paddle_posx = comp_paddle_posx / 80
        ball_velox = (ball_velox + 5) / 10
        ball_veloy = (ball_veloy + 5) / 10
        my_paddle_velox = (my_paddle_velox + 5) / 10
        comp_paddle_velox = (comp_paddle_velox + 5) / 10
                
        # create a new state
        state = np.array([[agent.new_experiences.ball_posx[-1], agent.new_experiences.ball_posy[-1], agent.new_experiences.ball_velox[-1], agent.new_experiences.ball_veloy[-1], agent.new_experiences.my_paddle_posx[-1], agent.new_experiences.my_paddle_velox[-1], agent.new_experiences.comp_paddle_posx[-1]]])
                
        # get next action, using next state
        next_action = agent.get_action(state)
               
        # if game is over, return the score and True
        if next_done:
            agent.new_experiences.add_experience(ball_posx, ball_posy, ball_velox, ball_veloy, my_paddle_posx, my_paddle_velox, comp_paddle_posx, next_action, next_reward + hit_ball_reward, next_done)
            return (score + next_reward), True
    
        # add the experience to the nex experience buffer    
        agent.new_experiences.add_experience(ball_posx, ball_posy, ball_velox, ball_veloy, my_paddle_posx, my_paddle_velox, comp_paddle_posx, next_action, next_reward + hit_ball_reward, next_done)
            
        # if enough data available in the experience buffer start to learn
        if len(agent.experiences.ball_posx) > agent.starting_mem_len:
            agent.learn()
    
        return (score + next_reward), False
    
    def _downgrade_frame(self, frame):
        '''
        publlic method to downgrade the frame
        frame: the  the list of rewards
        ''' 
        # cut the image - remove the frame
        frame = frame[35:195:2,::2,:]
        frame = np.average(frame, axis = 2)
        # remove the color
        frame = np.array(frame, dtype = np.uint8)
            
        return frame
   
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
    
class Agent():
    '''
    class of agent to learn playing pong using a CNN
    '''
    def __init__(self):
        '''
        class constructur
        ''' 
        self.experiences = Experiences(500000)          # all experiences
        self.new_experiences = New_Experiences()        # all experiences of a single episode
        self.possible_actions = [0,2,3]                 # all possible actions
        self.epsilon = 1                                # start value of
        self.epsilon_decay = .3/100000                  # delta reduction of epsilon
        self.epsilon_min = 0.05                         # minimum value of epsilon
        self.gamma = 0.95                               # gamma
        self.learn_rate = 0.00005                         # learn rate increased
        self.model = self._build_model()                # CNN learn model
        self.model_target = clone_model(self.model)     # target learn model
        self.total_timesteps = 0                        # total time steps
        self.starting_mem_len = 1000                    # start value to learn
        self.steps = 0                                  # learn steps
        self.batch_size = 32                            # batch size to fit neural network
        
    def _build_model(self):
        '''
        private method to create the models
        ''' 
        model = Sequential()
        model.add(Input((7,)))
        model.add(Dense(1024, activation = 'relu'))
        model.add(Dense(1024, activation = 'relu'))
        model.add(Dense(1024, activation = 'relu'))
        model.add(Dense(1024, activation = 'relu'))
        model.add(Dense(len(self.possible_actions), activation = 'sigmoid'))
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
        if self.experiences.done[index]:
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
            return random.sample(self.possible_actions,1)[0]

        # expedation: calcualate the best action
        a_index = np.argmax(self.model.predict(state, verbose=0))
        
        return self.possible_actions[a_index]
 
    def learn(self):
        '''
        public method to learn the CNN model
        ''' 
        # create a minibatch
        states = []                 # the states in the minibatch
        next_states = []            # the next states in the minibatch
        actions_taken = []          # the action take at states in the minibatch
        next_rewards = []           # the reward of states in the minibatch
        next_done_flags = []        # the done-flag in the minibatch
                
        while len(states) < self.batch_size:
        
            # take a random index of the experience queue
            index = np.random.randint(4,len(self.experiences.ball_posx) - 1)
            
            if self._index_valid(index):

                state = [self.experiences.ball_posx[index], self.experiences.ball_posy[index], self.experiences.ball_velox[index], self.experiences.ball_veloy[index], self.experiences.my_paddle_posx[index], self.experiences.my_paddle_velox[index], self.experiences.comp_paddle_posx[index]]
                next_state = [self.experiences.ball_posx[index+1], self.experiences.ball_posy[index+1], self.experiences.ball_velox[index+1], self.experiences.ball_veloy[index+1], self.experiences.my_paddle_posx[index+1], self.experiences.my_paddle_velox[index+1], self.experiences.comp_paddle_posx[index+1]]
                states.append(state)
                next_states.append(next_state)
                actions_taken.append(self.experiences.action[index])
                next_rewards.append(self.experiences.reward[index + 1])
                next_done_flags.append(self.experiences.done[index + 1])

        # calculate the output of the model and the target model to estimate the error function
        target = self.model.predict(np.array(states), verbose = 0)
        target_value = self.model_target.predict(np.array(next_states), verbose = 0)
        
        # define the labels to train the model
        for i in range(len(states)):
            action = self.possible_actions.index(actions_taken[i])
            target[i][action] = next_rewards[i] + (not next_done_flags[i]) * self.gamma * max(target_value[i])

        # train the model using the states and output
        self.model.fit(np.array(states), target,batch_size = self.batch_size, epochs = 1, verbose = 0)

        # reduce the epsilon for the exploration/expedetion trade-off
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
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

def play_game(game, agent):
    '''
    public method to play a game
    game: the game environment
    agent: the agent with the learning model
    ''' 
    # reset the environment
    game.reset_game()
    # create a start frame and a empty experience at the beginning
    agent.new_experiences.add_experience(0, 0, 0, 0, 0, 0, 0, 0, 0, False)
    
    # play a new game
    done = False
    score = 0
    while True:
        score, done = game.take_step(agent, score)
        if done:
            break
    return score

def main():         
    '''
    main function
    '''
    name = 'PongDeterministic-v4'
    game = GameEnvironment(name)
    agent = Agent()
    
    espisodes = 500
    scores = deque(maxlen = espisodes)
    max_score = -21
    
    # play 100 episodes
    for i in range(espisodes):
        
        timesteps = agent.total_timesteps
        timee = time.time()
        
        score = play_game(game, agent) 
         
        scores.append(score)
        if score > max_score:
            max_score = score
        
        # add the new experiences of the last episto the set of all experiences
        modified_reward = modify_reward(agent.new_experiences.reward)
            
        for j in range(len(agent.new_experiences.ball_posx)):
            
            next_ball_posx = agent.new_experiences.ball_posx[j]
            next_ball_posy = agent.new_experiences.ball_posy[j]
            next_ball_velox = agent.new_experiences.ball_velox[j]
            next_ball_veloy= agent.new_experiences.ball_veloy[j]
            next_my_paddle_posx = agent.new_experiences.my_paddle_posx[j]
            next_my_paddle_velox = agent.new_experiences.my_paddle_velox[j]
            next_comp_paddle_posx = agent.new_experiences.comp_paddle_posx[j]
            next_action = agent.new_experiences.action[j]
            next_reward = modified_reward[j]
            next_done = agent.new_experiences.done[j]
        
            if (next_ball_posx != 0 and next_ball_posy != 0 and next_ball_velox != 0 and next_ball_veloy != 0 and next_my_paddle_posx != 0 and next_comp_paddle_posx != 0):
        
                agent.experiences.add_experience(next_ball_posx, next_ball_posy, next_ball_velox, next_ball_veloy, next_my_paddle_posx, next_my_paddle_velox, next_comp_paddle_posx, next_action, next_reward, next_done)
            
        # delete the exeriences from the last episode
        agent.new_experiences.clear()
    
        print('\nepisode: ' + str(i))
        print('steps: ' + str(agent.total_timesteps - timesteps))
        print('duration: ' + str(time.time() - timee))
        print('score: ' + str(score))
        print('max. score: ' + str(max_score))
        print('epsilon: ' + str(agent.epsilon))
    
    plt.plot(scores)
    ipythondisplay.clear_output(wait=True)
    game.close_game()

'''
main
'''
if __name__ == '__main__':
    main()  
