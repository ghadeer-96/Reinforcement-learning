#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import pandas as pd


# * classes were built for path cells and exit cells
# * class was built for environment which have a map for rewards or for relation between states
#   that can give the possible next actions, if threre is any relation between 2 states
# * the agent class was built which will learn and explore the environment and exploite what he     learns  
#   the agent class create a table for Q-learn values (states as rows and actions as columns)
#   , the agent will update Qs during the exploration stage
# * the agent uses environment to select initial, next states and actions

# * number of exit states = 2*width - 2
# * number of path states = width - 2
# * number of states = 3*width - 4
# 
# * E >> current -1 
# * W >> current +1
# * N >> current -width
# * S >> current +width

# In[2]:



class PathCell():
    '''
    class to represent path cells with their actions and living reward and Q-learn values
    '''
    def __init__(self,livingReward):
self.possibleActions = ['E','W','N','S']
self.reward = livingReward
self.Q = [0,0,0,0]
self.shape = ' [WSNE] '


class ExitCell():
    '''
    class to represent exit cells with their actions and reward
    '''
    def __init__(self, reward=100):
self.possibleActions = ['exit']
self.reward = reward
self.shape = ' [exit] '


# In[3]:


class environment:
    '''
        class environment: 
        initiate the environment with states, living rewards, episodes, learning rate and discount
        * width: width of the grid in the environment
        * livingReward: living reward which agent take when go from path cell to another path cell
        * stateSpace: states space as a list of states, using integer values to reresent states
        * exitStates: list of exit states as idexsis
        * pathStates: list of path states as indesis
        * bigRewardStates: 2 random exit states with reward of +100
        * dictStatesObj: dictionary of objects of classes PathCell and ExitCell
        * Rs: numpy 2D array and dataframe (table) to represent relationship between states based on rewards
        * epoch: number of episodes
        * alpha: learning rate
        * gamma: discount value
    '''
    def __init__(self, livingReward,width, epoch, alpha, gamma):
        '''
        constructor to build the environment
        '''
        
        #width of the grid in the environment 
        self.width = width
        #living reward which agent take when go from path cell to another path cell
        self.livingReward = livingReward
        #states space as a list of states, using integer values to reresent states
        self.stateSpace = [i for i in range(3*self.width)]
        #exit states as indesis
        self.exitStates = []
        #path states as indesis
        self.pathStates = []
        #define path and exit states
        self.defStates()
        #random exit states with reward of +100
        self.bigRewardStates = self.select2positiveExitState()
        #dictionary of objects of classes PathCell and ExitCell
        self.dictStatesObj = dict()
        self.dictionaryOfStatesObj()
        
        #numpy 2D array to represent relationship between states based on rewards
        self.Rs = None #numpy 2D array
        self.df_Rs = None #Dataframe of Rs , used for debugging and tracing only
        
        
        #number of episodes
        self.epoch = epoch
        #learning rate
        self.alpha = alpha
        #discount value
        self.gamma = gamma

        
    def defStates(self):
        '''
        define and declare all states (exit and path) as numbers or indesis
        example: if width = 4 >> exit states:[1,2,4,7,9,10] path states:[5,6]
        '''
        exit1 = [i for i in range(1,self.width-1)] #cells in the first row
        exit2 = [self.width,2*self.width-1] #cells in the second row
        exit3 = [i for i in range(2*self.width+1,3*self.width-1)] #cells in the third row
        self.exitStates.extend(exit1)
        self.exitStates.extend(exit2)
        self.exitStates.extend(exit3) 
        for i in self.stateSpace:
            if i not in self.exitStates and i not in [0,self.width-1,len(self.stateSpace)-1,2*self.width]:
                self.pathStates.append(i)
        
        
    def select2positiveExitState(self):
        ''' select 2 exit states randomly to have +100 as reward'''
        states = random.choices(self.exitStates, k =2)
        while states[0] == states[1]:
            states = random.choices(self.exitStates, k =2)
        return states
    
    
    def dictionaryOfStatesObj(self):
        '''
        create objects for all cells according to thier types (exit or path), and 
        keep them in a dictionary{keys: indesis of states, values: (path or exit cell)}
        '''
        cell = None
        for i in self.stateSpace:
            if i in self.exitStates: # if it was exit state
                if i in self.bigRewardStates: # check if it will have a reward of 100
                    cell = ExitCell(100)
                else:
                    cell = ExitCell(-100)
                self.dictStatesObj[i] = cell
            if i in self.pathStates: # if it was path state
                cell = PathCell(self.livingReward)
                self.dictStatesObj[i] = cell
        
        
    def buildRs(self):
        '''
        build living reward table
        this function build relation ships between states based on rewards
        so if the reward in an entry wasn't None or 0 then we can walk from 
        our current state to this rekated state
        '''
        self.Rs = np.zeros([len(self.stateSpace),len(self.stateSpace)])
        
    
        for i in range(0,self.Rs.shape[0]): #rows
            for j in range(0,self.Rs.shape[1]): #cols
                if i == j+1 or i == j-1:
                    self.Rs[i][j] = self.livingReward
        for i in range(0,self.Rs.shape[1]):
            try:
                self.Rs[i][i+self.width] = self.livingReward
                self.Rs[i+self.width][i] = self.livingReward 
            except IndexError as error:
                continue
            
        for i in range(0,self.Rs.shape[0]):
            try:
                for j in [i+self.width,i-self.width,i+1,i-1]:
                    if j in self.exitStates and j in self.bigRewardStates:
                        self.Rs[i][j] = 100
                    elif j in self.exitStates and j not in self.bigRewardStates:
                        self.Rs[i][j] = -100
            except IndexError as error:
                continue            
                    
        for i in range(0,self.width): #first set of rows
            for j in range(0,self.Rs.shape[1]): #all cols
                self.Rs[i][j] = None
        for i in range(2*self.width,len(self.stateSpace)-1): #last set of rows
            for j in range(0,self.Rs.shape[1]): #all cols
                self.Rs[i][j] = None
                
        for i in [0,self.width-1,len(self.stateSpace)-1,2*self.width]: #empty states or squares in the end
            self.Rs[i] = None
            self.Rs[:,i] = None


        for i in range(0,self.Rs.shape[0]): # diagonal
            self.Rs[i][i] = None
           
            
        for i in range(0,self.Rs.shape[0]):
            for j in range(0,self.Rs.shape[1]):#cols
                if self.Rs[i][j] ==0 :
                    self.Rs[i][j]=None
                
        self.Rs[self.width,:] = None
        self.Rs[2*self.width-1,:] = None  
        
        
        self.df_Rs = pd.DataFrame(data=self.Rs[0:,0:])
        

    def render(self):
        '''
        this function to render the environment 
        '''
        # render the environment
        str = [[],[],[]]
        for v in self.dictStatesObj.keys():
            if v in range(1,self.width-1):
                str[0].append(self.dictStatesObj[v].shape)
                
            if v in range(2*self.width+1,3*self.width-1):
                str[2].append(self.dictStatesObj[v].shape)
            else:
                if isinstance(self.dictStatesObj[v], PathCell):
                    str[1].append(self.dictStatesObj[v].Q)
                if v in [self.width,2*self.width-1]:
                    str[1].append(self.dictStatesObj[v].shape)

                    
        for i in range(len(str)):        
            print(str[i])
            
        


        
          


# In[ ]:





# In[4]:


class agent:
    '''
    * env: the environment where the agent will learn, explore and exploite  
    * possibleActions: all possible actions for a path cell
    * Qs: numpy 2D array and dataframe (table) to represent all states with their related Q-learning values
    * currentState: index of the current state which the agent is in    
    '''
    def __init__(self, env):
        self.env = env
        #all possible actions for a path cell
        self.possibleActions = ['E','W','N','S']
        #numpy 2D array and dataframe (table) to represent all states with their related Q-learning values
        self.Qs = None
        self.df_Qs = None
        self.buildQs()
        #index of the current state which the agent is in
        self.currentState = None  #index of current state like 4 >> state5
        self.nextState = None
        
        
    
    def selectInitialState(self):
        '''
        this function randomly choose initial state (from path cells) to start with
        '''
        self.currentState = random.choice(self.env.pathStates)
        return self.currentState
    
    def isExitState(self, state):
        '''
        this function check if the current state is an exit state or not
        '''
        if state in self.env.exitStates:
            return True
        return False
    
    def selectRandomAction(self):
        '''
        this function randomly choose action from the possible action of the current state
        '''
        return random.choice(self.env.dictStatesObj[self.currentState].possibleActions)
    
    def selectNextState(self):
        '''
        this function randomly next state for the current state, taking consideration the
        current state and its possible actions.
        first >> check all possible next states
        second >> choose random action from the possible actions
        third >> select random next state from the possible next states
        forth >> update Q-learn values for current state
        '''
        nextState = None  
        current = self.currentState 
        
        LR = env.livingReward
        states = []
        # get all possible next states
        for col in range(0,env.Rs.shape[1]):
            if env.Rs[current][col] == LR or env.Rs[current][col] == 100 or env.Rs[current][col] == -100 :
                states.append(col)
        print(states)
        # select random action
        action = self.selectRandomAction()
        print('action = ',action)
        #select random next state
        next_state_prob = random.random()
        if next_state_prob >0 and next_state_prob <=0.7:
            nextState = self.chooseNextState(action)
        if next_state_prob >0.7 and next_state_prob <=0.8:
            nextState = random.choice(states)
        if next_state_prob >0.8 and next_state_prob <=0.9:
            nextState = random.choice(states)
        if next_state_prob >0.9 and next_state_prob <=1:
            nextState = random.choice(states)
        self.nextState = nextState
        print('next state: ',self.nextState)
        self.updateQs(self.currentState, action, self.nextState)
        
        
    def chooseNextState(self, action):
        '''
        choose next state with certainty 
        '''
        if action == 'E': # east then go to next cell in same row
            return self.currentState +1
        if action == 'W': # west then go to previos cell in same row
            return self.currentState -1
        if action == 'N': # north then go to previos cell in the same column
            return self.currentState -self.env.width
        if action == 'S': # south then go to next cell in the same column
            return self.currentState +self.env.width
    
    def buildQs(self):
        '''
        build Q-learn values table with rows as states and columns as actions
        all values initially are 0
        for states or cells which are not exit or path cells, Q is None 
        Q-learn for exit states is not used so it is kept as 0
        '''
        self.Qs = np.zeros([len(self.env.stateSpace),len(self.possibleActions)])
        for i in [0,self.env.width-1,len(self.env.stateSpace)-1,2*self.env.width]:
            self.Qs[i,:] = None
        self.df_Qs = pd.DataFrame(data=self.Qs[0:,0:],columns = self.possibleActions)
        
        
    def updateQs(self, state, action, new_state):
        '''
        update Q-learn values for a state with take in consideration the sction and the new state
        reward, alpha, gamma and the old Q-learn for the current and the next states  
        '''
        if action == 'E':
            action = 0
        if action == 'W':
            action = 1
        if action == 'N':
            action = 2
        if action == 'S':
            action = 3
        reward = self.env.Rs[state,new_state]
        self.Qs[state,action] = self.Qs[state,action] + self.env.alpha * (reward + self.env.gamma * np.max(self.Qs[new_state,:]) - self.Qs[state,action])
        # update Qs for cells in the dictionary
        self.env.dictStatesObj[state].Q = self.Qs[state,:]
        
        
        
    def explore_exploite(self):
        '''
        this function will explore and learn then go to exploite and use what is learned
        '''
        epoch = self.env.epoch        
        alpha = self.env.alpha
        
        #select initial state randomly
        self.selectInitialState()
        initial_state = self.currentState
        #for explore and learn:
        #calcultae the number of episodes for exploration 
        explore_epoch = int(epoch * alpha)
        print(explore_epoch)
        
        # learning stage
        for i in range(0,explore_epoch):
            print('iteration: ',i)
            print('initial state: ',initial_state)
            # check if the current state is exit state to exit this episode 
            while(self.isExitState(self.currentState) == False):
                #select random action and next state
                self.selectNextState()
                #update current state to be the next state
                self.currentState = self.nextState
            #change current state to be the initial again
            self.currentState = initial_state
        
        #for exploite 
        #calcultae the number of episodes for exploitation
        exploit_epoch = int(epoch - (explore_epoch))
        #
        self.selectInitialState()
        initial_state = self.currentState
        # this list will keep all episodes we got 
        step = []
        
        
        for i in range(0,exploit_epoch ):
            print('iteration: ',i)
            print('initial state: ',initial_state)
            
            # check if the current state is exit state to exit this episode
            while(self.isExitState(self.currentState) == False):
                #it is exploitation so we have to get the action that give us max Q-learn value
                action = np.argmax(self.Qs[self.currentState,:])
                if action == 0:
                    actionName = 'E'
                if action == 1:
                    actionName = 'W'
                if action == 2:
                    actionName = 'N'
                if action == 3:
                    actionName = 'S'
                step.append('C'+str(self.currentState))
                #step.append(self.env.dictStatesObj[self.currentState].shape)
                step.append(actionName)
                #according to the action we choose a next state based on the policy (certainty)
                self.nextState = self.chooseNextState(actionName)
                step.append('C'+str(self.nextState))
                #update current state to be the next state
                self.currentState = self.nextState
                
                print(step)
                #delete last printed episode 
                step = []
            #change current state to be the initial again
            self.currentState = initial_state
 
        print(exploit_epoch)


# In[5]:


env = environment(6,4,100,0.8,0.5)
env.buildRs()
env.render()


# In[6]:


for i in env.stateSpace:
    if i in env.pathStates or i in env.exitStates:
        print(i,env.dictStatesObj[i].reward)


# In[7]:


env.df_Rs


# In[ ]:





# In[8]:


a = agent(env)


# In[9]:


a.explore_exploite()


# In[10]:


a.df_Qs


# In[11]:


a.env.render()


# In[ ]:





# In[ ]:





# In[ ]:




