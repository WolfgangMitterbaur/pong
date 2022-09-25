# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 20:39:44 2022

@author: Wolfgang MItterbaur

file name solutionq.py

one step Q-learning to learn playing pong
"""

# all imports
import numpy as np
import gym
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
    
    def __init__(self,max_len):
        '''
        class constructur
        '''
        self.max_len = max_len                              # maximal length of double-ended queue
        self.ball_posx = deque(maxlen = max_len)            # queue of ball position x
        self.ball_posy = deque(maxlen = max_len)            # queue of ball position y
        self.ball_movex = deque(maxlen = max_len)           # queue of ball movements in x
        self.ball_movey = deque(maxlen = max_len)           # queue of ball movements in y
        self.my_paddle_posx = deque(maxlen = max_len)       # queue of my paddle position x
        self.my_paddle_movex = deque(maxlen = max_len)      # queue of my paddle movemement in x
        self.comp_paddle_posx = deque(maxlen = max_len)     # queue of computer´s paddle position in x
        self.action = deque(maxlen = max_len)               # queue of taken actions
        self.reward = deque(maxlen = max_len)               # queue of received reward of next state
        self.done = deque(maxlen = max_len)                 # queue of done-flags 
        

    def add_experience(self, ball_posx, ball_posy, ball_movex, ball_movey, my_paddle_posx, my_paddle_movex, comp_paddle_posx, action, reward, done):
        '''
        public method to add one experience
        ball_posx: the position x of the ball
        ball_posy: the position y of the ball
        ball_movex: the movement in x of the ball
        ball_movey: the movements in y of the ball
        my_paddle_posx: the position x of my paddle
        my_paddle_movex: the movement of my paddle
        comp_paddle_posx: the position x of the computer´s paddle
        action: next taken action 
        reward: reward received from the environment
        done: next done-flag
        '''
        self.ball_posx.append(ball_posx)
        self.ball_posy.append(ball_posy)
        self.ball_movex.append(ball_movex)
        self.ball_movey.append(ball_movey)
        self.my_paddle_posx.append(my_paddle_posx)
        self.my_paddle_movex.append(my_paddle_movex)
        self.comp_paddle_posx.append(comp_paddle_posx)
        self.action.append(action)
        self.reward.append(reward)
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
        self.ball_posx = []                     # list of all x positions of the ball
        self.ball_posy = []                     # list of all y positions of the ball
        self.ball_movex = []                    # list of all movements of the ball in x direction
        self.ball_movey = []                    # list of all movements of the ball in y direction
        self.my_paddle_posx = []                # list of all x positions of my paddle
        self.my_paddle_movex = []               # list of all movements of my paddle in x direction
        self.comp_paddle_posx =  []             # list of all x positions of the computer´s paddle
        self.action = []                        # list of all actions
        self.reward = []                        # list of all rewards
        self.done = []                          # list of all done flags

    def add_experience(self, ball_posx, ball_posy, ball_movex, ball_movey, my_paddle_posx, my_paddle_movex, comp_paddle_posx, action, reward, done):
        '''
        public method to add one experience
        ball_posx: next ball x position to add
        ball_posy: next ball y position to add
        ball_movex: the next ball movement in x direction to add
        ball_movey: the next ball movemetn in y direction to add
        my_paddle_posx: the next x position of my paddle to add
        my_paddle_movex: the next movement in x direction of my paddle to add
        comp_paddle_posx: the next x position of the computer´s paddle to add
        action: the next action to add
        reward: the next reward to add
        done: the next done-flag to add
        '''
        self.ball_posx.append(ball_posx)
        self.ball_posy.append(ball_posy)
        self.ball_movex.append(ball_movex)
        self.ball_movey.append(ball_movey)
        self.my_paddle_posx.append(my_paddle_posx)
        self.my_paddle_movex.append(my_paddle_movex)
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
        self.ball_movex = []
        self.ball_movey = []
        self.my_paddle_posx = []
        self.my_paddle_movex = []
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
        self.lst_ball_posx = deque(maxlen = 2)              # list of the ball positions in x
        self.lst_ball_posy = deque(maxlen = 2)              # list of the ball positions in y
        self.lst_my_paddle_posx = deque(maxlen = 2)         # list of my paddle positions in x
        self.lst_comp_paddle_posx = deque(maxlen = 2)       # list of the computer´s positions in x
        self.hit_ball = False                               # flag when the paddle hit the ball
        self.ball_posx = 0                                  # the position x of the ball
        self.ball_posy = 0                                  # the position y of the ball
        self.my_paddle_pos_min = 0                          # the minimum position x of my paddle
        self.my_paddle_pos_max = 0                          # the maximum position y of my paddle
        self.comp_paddle_pos_min = 0                        # the minimum position x of the computer paddle
        self.comp_paddle_pos_max = 0                        # the maximum position x of the computer paddle
        self.ball_movex = 0                                 # the movement of the ball in x
        self.ball_movey = 0                                 # the movement of the ball in y
        self.my_paddle_movex = 0                            # the movement of my paddle in x
        self.com_paddle_movex = 0                           # the movement of the computer´s paddle in x
        
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
        public method to calculate positions of the ball, my paddle, computer paddle at a given frame
        frame: the frame to be analyzed
        ''' 
        self.ball_posx = 0                                  # the position x of the ball
        self.ball_posy = 0                                  # the position y of the ball
        self.my_paddle_posx_min = 0                         # the minimum position x of my paddle
        self.my_paddle_posx_max = 0                         # the maximum position y of my paddle
        self.comp_paddle_posx_min = 0                       # the minimum position x of the computer paddle
        self.comp_paddle_posx_max = 0                       # the maximum position x of the computer paddle
            
        # loop over all lines (x-coordinates)
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
        
        
        my_paddle_posx_middle = (self.my_paddle_posx_min + self.my_paddle_posx_max)/2           # middle position of paddle
        comp_paddle_posx_middle = (self.comp_paddle_posx_min + self.comp_paddle_posx_max)/2     # middle position of paddle
    
        # add the positions to the queue
        self.lst_ball_posx.append(self.ball_posx)
        self.lst_ball_posy.append(self.ball_posy)
        self.lst_my_paddle_posx.append(my_paddle_posx_middle)
        self.lst_comp_paddle_posx.append(comp_paddle_posx_middle)
        
        return self.ball_posx, self.ball_posy, my_paddle_posx_middle, comp_paddle_posx_middle
    
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
            my_paddle_velox = self.lst_my_paddle_posx[1] - self.lst_my_paddle_posx[0]
            
            if my_paddle_velox > 0:
                self.my_paddle_movex = 2
            elif my_paddle_velox < 0:
                self.my_paddle_movex = 1
            else:
                self.my_paddle_movex = 0  
            
        # calculate the movement using the last two positions
        self.comp_paddle_movex = 0 
        if len(self.lst_comp_paddle_posx) == 2:
            comp_paddle_velox = self.lst_comp_paddle_posx[1] - self.lst_comp_paddle_posx[0]
            
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
        # check if my paddle hit the ball
        hit_ball_reward = 0
    
        if self.ball_posy < 60:
            self.hit_ball = False
        
        if self.ball_posy <= 69 and self.ball_posy >= 68 and not self.hit_ball:
            
            if (self.my_paddle_pos_min - 1 <= self.ball_posx and self.my_paddle_pos_max + 1 >= self.ball_posx)\
                or (self.my_paddle_pos_min - 2 <= self.ball_posx and self.my_paddle_pos_min + 1 >= self.ball_posx and self.my_paddle_movex == 1)\
                or (self.my_paddle_pos_max + 2 >= self.ball_posx and self.my_paddle_pos_max - 1 <= self.ball_posx and self.my_paddle_movex == 2):
                
                self.hit_ball = True            
                hit_ball_reward = 0.3                                       # a small positive reward if the my paddle hit the ball              

        return hit_ball_reward
    
class GameEnvironment(object):
    '''
    class of the game environment
    '''
    
    def __init__(self, name):
        '''
        class constructur
        '''
        self.analyse = AnalyseFrame()                                       # instance to analyse the frame         
        self.Q = np.zeros(shape=(40, 40, 3, 3, 40, 3), dtype = 'float')     # Q-value table
        self.experiences = Experiences(500000)                              # all experiences
        self.new_experiences = New_Experiences()                            # the new experiences of a new episode
        self.alpha = 0.1                                                    # learning rate alpha
        self.epsilon = 1                                                    # start value for the exploration
        self.epsilon_decay = 1/10000000                                     # decreasing value for the exploration
        self.epsilon_min = 0.05                                             # the minimum value for the exploration
        self.gamma = 0.4                                                    # the discount factor
        self.possible_actions=[0, 2, 3]                                     # the possible actions of the game
        self.steps = 0                                                      # number of al learning steps
        self.env = self._make_env(name)
        
        # initialize the Q table
        for a in range(40):
            for b in range(40):
                for c in range(3):
                    for d in range(3):
                        for e in range(40):
                            for f in range(3):
                    
                                self.Q[a][b][c][d][e][f] = (-0.7+0.9)*np.random.sample() - 0.9

    def _make_env(self, name):
        '''
        private method to create a new game environment
        name: the name of the game
        '''
        #env = gym.make(name, render_mode='human        # render mode to view the game environment
        env = gym.make(name)
        return env

    def _take_step(self, score):
        '''
       private method to play the next step
       env: the environment of the game
       score: the current score of this episode
       '''
          
        # perform a step on the environment
        next_frame, next_reward, next_done, info = self.env.step(self.new_experiences.action[-1])
            
        # resize the frame of the next step
        next_frame = self._downgrade_frame(next_frame)
         
        # analyse the frame        
        ball_posx, ball_posy, my_paddle_posx, comp_paddle_posx = self.analyse.calc_positions(next_frame)
        ball_movex, ball_movey, my_paddle_movex, comp_paddle_movey = self.analyse.calc_movements()
       
        hit_ball_reward = self.analyse.ball_hit_reward()
        
        # reduce the size of the obtained values 
        new_ball_posx_r = np.byte(ball_posx/2)
        new_ball_posy_r = np.byte(ball_posy/2)
        new_ball_movex_r = np.byte(ball_movex)
        new_ball_movey_r = np.byte(ball_movey)
        new_my_paddle_posx_r = np.byte(my_paddle_posx/2)
        new_my_paddle_movex_r = np.byte(my_paddle_movex)
        new_comp_paddle_posx_r = np.byte(comp_paddle_posx/2)
       
        # select a new action
        if np.random.rand() < self.epsilon:
            
            # take a random action
            new_action = random.sample(self.possible_actions,1)[0]
        
        else:
            
            # take the action depending on the highest q-value
            Q1 = self.Q[new_ball_posx_r][new_ball_posy_r][new_ball_movex_r][new_ball_movey_r][new_my_paddle_posx_r][0]  # action 0 = stay
            Q2 = self.Q[new_ball_posx_r][new_ball_posy_r][new_ball_movex_r][new_ball_movey_r][new_my_paddle_posx_r][1]  # action 2 = up
            Q3 = self.Q[new_ball_posx_r][new_ball_posy_r][new_ball_movex_r][new_ball_movey_r][new_my_paddle_posx_r][2]  # action 3 = down
        
            if Q2 > Q1 and Q2 > Q3:
                new_action = self.possible_actions[1]
            elif Q3 > Q1 and Q3 > Q2:
                new_action = self.possible_actions[2]
            else:
                new_action = self.possible_actions[0]
        
        reward =  next_reward + hit_ball_reward
        
        # add the reward to the queue
        self.new_experiences.add_experience(new_ball_posx_r, new_ball_posy_r ,new_ball_movex_r, new_ball_movey_r, new_my_paddle_posx_r, new_my_paddle_movex_r, new_comp_paddle_posx_r, new_action, reward, next_done)        
                        
        # learn the new Q-values - take random 8 experiences from the experiences relay to increase the learining performance
        if self.steps > 2000:                  
            
            for i in range(8):
                
                # take a random experience from the buffer
                index = np.random.randint(2, len(self.experiences.ball_posx) - 2)
                
                # state
                learn_ball_posx_r = self.experiences.ball_posx[index-2]
                learn_ball_posy_r = self.experiences.ball_posy[index-2]
                learn_ball_movex_r = self.experiences.ball_movex[index-2]
                learn_ball_movey_r = self.experiences.ball_movey[index-2]
                learn_my_paddle_posx_r = self.experiences.my_paddle_posx[index-2]
                learn_action_r_no = self.experiences.action[index-2]
                
                if learn_action_r_no == 2:
                    learn_action_r = 1
                elif learn_action_r_no == 3:
                    learn_action_r = 2
                else:
                    learn_action_r = 0
                
                learn_reward = self.experiences.reward[index-2]
                
                #new_state
                learn_next_ball_posx_r = self.experiences.ball_posx[index]
                learn_next_ball_posy_r = self.experiences.ball_posy[index]
                learn_next_ball_movex_r = self.experiences.ball_movex[index]
                learn_next_ball_movey_r = self.experiences.ball_movey[index]
                learn_next_my_paddle_posx_r = self.experiences.my_paddle_posx[index]
                
                maxQnew = np.max(self.Q[learn_next_ball_posx_r][learn_next_ball_posy_r][learn_next_ball_movex_r][learn_next_ball_movey_r][learn_next_my_paddle_posx_r])
             
                self.Q[learn_ball_posx_r][learn_ball_posy_r][learn_ball_movex_r][learn_ball_movey_r][learn_my_paddle_posx_r][learn_action_r] = \
                   self.Q[learn_ball_posx_r][learn_ball_posy_r][learn_ball_movex_r][learn_ball_movey_r][learn_my_paddle_posx_r][learn_action_r] +\
                   self.alpha * ( learn_reward + self.gamma * maxQnew - self.Q[learn_ball_posx_r][learn_ball_posy_r][learn_ball_movex_r][learn_ball_movey_r][learn_my_paddle_posx_r][learn_action_r])
   
                
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
            
        self.steps += 1
        
        # if game is over, return the score
        if next_done:
            return (score + next_reward), True
    
        return (score + next_reward), False
    
    def _downgrade_frame(self, frame):
        '''
        private method to downgrade the frame
        frame: the  the list of rewards
        '''
        frame = frame[35:195:2, ::2, :]                 # cut the image - remove the frame
        frame = np.average(frame, axis=2)
        frame = np.array(frame, dtype=np.uint8)         # remove the color

        return frame
    
    def play_game(self):
        '''
        public method to play a game
        game: the game environment
        env: the game environment
        '''
        # reset the environment
        self.env.reset()
        # create a start frame and a empty experience at the beginning
        self.new_experiences.add_experience(0, 0, 0, 0, 0, 0, 0, 0, 0, False)
        
        # play a new game
        done = False
        score = 0
        while True:
            score, done = self._take_step(score)
            if done:
                break
        return score
    
    def close(self):
        '''
        public method to close the game
        '''
        self.env.close()
        
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
         
def main(): 
    '''
    main function to start the program
    generates the game environment and the agent and start all episodes
    '''
    name = 'PongDeterministic-v4'
    game = GameEnvironment(name)
    episodes = 0                                # number of episodes
    scores = deque(maxlen = 100)
    max_score = -21
    score = -21
        
    while (score < 15):
        
        episodes += 1
        
        time_act = time.time()
        score = game.play_game() #set debug to true for rendering
      
        scores.append(score)
        
        if score > max_score:
            max_score = score
        
        # add the new experiences of the last episto the set of all experiences
        modified_reward = modify_reward(game.new_experiences.reward)
            
        for j in range(len(game.new_experiences.reward)):
        
            next_ball_posx = game.new_experiences.ball_posx[j]
            next_ball_posy = game.new_experiences.ball_posy[j]
            next_ball_movex = game.new_experiences.ball_movex[j]
            next_ball_movey = game.new_experiences.ball_movey[j]
            next_my_paddle_posx = game.new_experiences.my_paddle_posx[j]
            next_my_paddle_movex = game.new_experiences.my_paddle_movex[j]
            next_comp_paddle_posx = game.new_experiences.comp_paddle_posx[j]
            next_action = game.new_experiences.action[j]
            next_reward = modified_reward[j]
            next_done = game.new_experiences.done[j]
            
            if next_ball_posx != 0 and next_ball_posy != 0 and next_my_paddle_posx != 0 and next_comp_paddle_posx != 0:
                game.experiences.add_experience(next_ball_posx, next_ball_posy, next_ball_movex,  next_ball_movey, next_my_paddle_posx, next_my_paddle_movex, next_comp_paddle_posx, next_action, next_reward, next_done)
            
        # delete the exeriences from the last episode
        game.new_experiences.clear()
        
        print('\nEpisode: ' + str(episodes))
        print('Steps: ' + str(game.steps))
        print('Duration: ' + str(time.time() - time_act))
        print('Score: ' + str(score))
        print('Max Score: ' + str(max_score))
        print('Epsilon: ' + str(game.epsilon))
    
    plt.plot(scores)
    ipythondisplay.clear_output(wait=True)
    game.close()

'''
main
'''
if __name__ == '__main__':
    main()  
