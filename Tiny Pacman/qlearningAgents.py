# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from DQN import *
from collections import deque

import random,util,math
import matplotlib.pyplot as plt

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.QValues = util.Counter() #indexed by state and action

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.QValues[state, action]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state)
        if 'Stop' in legal_actions:
            legal_actions.remove('Stop')

        values = [self.getQValue(state, action) for action in legal_actions]
        if (values):
            return max(values)
        else:
            return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state) #all the legal actions

        value = self.getValue(state)
        for action in legal_actions:
            if (value == self.getQValue(state, action)):
                return action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if (util.flipCoin(self.params['eps'])):
            action = random.choice(legalActions)
            self.last_move = 'Random'
            while action == 'Stop':
                action = random.choice(legalActions)
                self.last_move = 'Random'
        else:
            action = self.getPolicy(state)
            self.last_move = 'Policy'

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        newQValue = (1 - self.alpha) * self.getQValue(state, action) #new Qvalue
        newQValue += self.alpha * (reward + (self.discount * self.getValue(nextState)))
        self.QValues[state, action] = newQValue

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', load_name = None, **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

        self.params = {

            #Model saves
            'load_file': None,
            'save_file': None,
            'save_interval': 20,

            'training_start': 100, # Number of steps to run before training of DQN starts
            'batch_size': 8, #Replay memory batch size
            'memory_size': 10000, #Size of replay memory
            'discount': self.discount,  # Discount rate (gamma value)
            'lr': .0001,  # Learning reate
            'rms_decay': 0.9,  # RMS Prop decay
            'rms_eps': 1e-8,  # RMS Prop epsilon

            # Epsilon value (epsilon-greedy)
            'eps': 1,  # Epsilon start value
            'eps_final': .1,  # self.epsilon,  # Epsilon end value
            'eps_step': 5000, #self.numTraining*10,  # Epsilon steps between start and end (linear)
            'eps_decay': 0.999,

            #DQN parameters
            'input_shape': 36, #Number of features in the state
            'hidden_nodes': 128,
            'number_actions': 4, #Number of possible actions
            'no_frames': 1,

            'window_size_H': 2*7-1,
            'window_size_W': 2*7-1
        }

        if load_name is not None:
            self.params['load_file'] = load_name
            self.params['save_file'] = None
            self.params['training_start'] = 0
            self.params['eps'] = 0

        # Initialise DQN
        self.qnet = DQN(self.params)
        #print(self.qnet.model.summary())
        self.qnet_target = DQN(self.params)

        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())

        # Initialise replay memory
        self.replay_mem = deque()
        self.replay_mem_special = deque()

        # Counter for number of steps taken
        self.local_cnt = 0
        self.cnt = 0
        self.cost_total = 0
        self.cnt_last = 0

        # Frame storing
        self.images = deque(maxlen=self.params['no_frames'])
        self.images_blend = self.params['no_frames']


    def getWeights(self):
        return self.weights

    def actionToInt(self,action):

        actionDict = {'North':0,'South':1,'East':2,'West':3,'Stop':4}
        return actionDict[action]

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        state_features = self.featExtractor.getStateVector_window(state,'P', self.params['window_size_W'],self.params['window_size_H'])
        #state_features = self.featExtractor.getStateVector(state)
        #state_features = self.featExtractor.getStateMatrices(state)

        """
        if len(self.images) == 0:
            self.images.append(self.featExtractor.getStateMatrices(state))

        state_features = self.blend_images(self.images, self.images_blend)
        """

        action_no = self.actionToInt(action)

        state_only = state_features[:-5].reshape(1,self.params['window_size_H'],self.params['window_size_W'],1)
        power_only = state_features[-5].reshape(1,)
        x_self = state_features[-4].reshape(1,)
        y_self = state_features[-3].reshape(1,)
        x_other = state_features[-2].reshape(1,)
        y_other = state_features[-1].reshape(1,)

        #QValue = self.qnet.model.predict(state_features.reshape(1,-1))
        QValue = self.qnet.model.predict([state_only,power_only,x_self,y_self,x_other,y_other])
        QValue = QValue[0][action_no]

        return QValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        self.local_cnt += 1
        #print(self.local_cnt)

        #self.last_state = self.featExtractor.getStateMatrices(state)
        self.last_state = self.featExtractor.getStateVector_window(state, 'P', self.params['window_size_W'],self.params['window_size_H'])

        #self.images.append(self.last_state)
        #self.last_state = self.blend_images(self.images, self.images_blend)

        self.last_action = self.actionToInt(action)
        self.last_reward = reward
        #self.current_state = self.featExtractor.getStateMatrices(nextState)
        self.current_state = self.featExtractor.getStateVector_window(nextState, 'P', self.params['window_size_W'],self.params['window_size_H'])

        #Check whether next state is terminal state
        self.terminal_state = False
        if len(self.getLegalActions(nextState)) == 0:
            self.terminal_state = True

        ###########################################################
        # Change rewards
        if reward > 20:
            self.last_reward = 50.    # Eat ghost   (Yum! Yum!)
        elif reward > 0:
            self.last_reward = 10.    # Eat food    (Yum!)
        elif reward < -10:
            self.last_reward = -500.  # Get eaten   (Ouch!) -500
            self.won = False
        elif reward < 0:
            self.last_reward = -1.    # Punish time (Pff..)
        elif reward > 400:
            self.last_reward = 100.

        ########################################################

        #Store experience in memory
        experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state,self.terminal_state)
        self.replay_mem.append(experience)

        # Remove oldest experience if replay memory full
        if len(self.replay_mem) > self.params['memory_size']:
            self.replay_mem.popleft()

        # Update epsilon
        if self.episodesSoFar < self.numTraining:
            self.params['eps'] = max(self.params['eps_final'], 1*self.params['eps_decay']**self.episodesSoFar)
        else:
            self.params['eps'] = 0


        # Save model
        if (self.params['save_file']):
            if (self.episodesSoFar % self.params['save_interval'] == 0) and (self.terminal_state):
                layout_name = state.data.layout.__dict__['name']
                name = layout_name
                save_location = 'saves/' + name + "_" + str(self.episodesSoFar)
                self.qnet.model.save(save_location)


        # Train model
        if (self.beta == 0):
            if self.episodesSoFar < self.numTraining:
                self.train()



    #Training
    def train(self):

        def TDerror(self,batch_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o,batch_a,batch_r,batch_t,batch_p_s):

            # Calculate TD error:
            q_vals = self.qnet.model.predict([batch_s,batch_p_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o])
            prediction = np.zeros(self.params['batch_size'],)
            target = np.zeros(self.params['batch_size'],)
            for r in range(self.params['batch_size']):
                prediction[r] = q_vals[r][batch_a[r]]
                if batch_t[r] == False:
                    target[r] = batch_r[r] + self.discount*q_vals_next[r]
                else:
                    target[r] = batch_r[r]

            error = np.sum(np.abs(target-prediction))

            return error


        # Start training if the minimum number of steps have been taken
        if (self.local_cnt > self.params['training_start']):# and (len(self.replay_mem_special) > self.params['batch_size']):
            # Prepare batch to sample from for training
            batch = random.sample(self.replay_mem, 2*self.params['batch_size'])
            #batch_special = random.sample(self.replay_mem_special, self.params['batch_size'])
            batch_s = []  # States (s)
            batch_r = []  # Rewards (r)
            batch_a = []  # Actions (a)
            batch_n = []  # Next states (s')
            batch_t = []  # Terminal state (t)
            batch_p_s = [] # Power pellet active current state (s)
            batch_p_n = [] # Power pellet active next state (n)
            batch_x_s = []
            batch_y_s = []
            batch_x_o = []
            batch_y_o = []
            batch_x_s_n = []
            batch_y_s_n = []
            batch_x_o_n = []
            batch_y_o_n = []

            for i in batch:
                batch_s.append(i[0][:-5].reshape(self.params['window_size_H'],self.params['window_size_W'],1))
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3][:-5].reshape(self.params['window_size_H'],self.params['window_size_W'],1))
                batch_t.append(i[4])
                batch_p_s.append(i[0][-5])
                batch_x_s.append(i[0][-4])
                batch_y_s.append(i[0][-3])
                batch_x_o.append(i[0][-2])
                batch_y_o.append(i[0][-1])
                batch_p_n.append(i[3][-5])
                batch_x_s_n.append(i[3][-4])
                batch_y_s_n.append(i[3][-3])
                batch_x_o_n.append(i[3][-2])
                batch_y_o_n.append(i[3][-1])

            """
            for i in batch_special:
                batch_s.append(i[0][:-1].reshape(self.params['window_size_H'],self.params['window_size_W'],1))
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3][:-1].reshape(self.params['window_size_H'],self.params['window_size_W'],1))
                batch_t.append(i[4])
                batch_p_s.append(i[0][-1])
                batch_p_n.append(i[3][-1])
            """

            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = np.array(batch_a)#self.get_onehot(np.array(batch_a))
            batch_n = np.array(batch_n)
            batch_t = np.array(batch_t)
            batch_p_s = np.array(batch_p_s)
            batch_x_s = np.array(batch_x_s)
            batch_y_s = np.array(batch_y_s)
            batch_x_o = np.array(batch_x_o)
            batch_y_o = np.array(batch_y_o)
            batch_p_n = np.array(batch_p_n)
            batch_x_s_n = np.array(batch_x_s_n)
            batch_y_s_n = np.array(batch_y_s_n)
            batch_x_o_n = np.array(batch_x_o_n)
            batch_y_o_n = np.array(batch_y_o_n)

            # Predicted Qvalues and NextQvalues
            q_vals = self.qnet.model.predict([batch_s,batch_p_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o])
            q_vals_next = self.qnet_target.model.predict([batch_n,batch_p_n,batch_x_s_n,batch_y_s_n,batch_x_o_n,batch_y_o_n])
            q_vals_next = np.max(q_vals_next,axis=1)

            for i in range(len(batch_a)):
                if batch_t[i] == False:
                    q_vals[i][batch_a[i]] = batch_r[i] + self.discount*q_vals_next[i]
                else:
                    q_vals[i][batch_a[i]] = batch_r[i]

            #error_before = TDerror(self,batch_s,batch_a,batch_r,batch_t)

            #feed_dict = {self.power: batch_p}
            self.qnet.model.fit([batch_s,batch_p_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o], q_vals, shuffle=True, epochs=10, batch_size=len(batch_s),verbose=0)

            error = TDerror(self,batch_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o,batch_a,batch_r,batch_t,batch_p_s)

            #print([error_before,error])

            cost = error
            self.cost_total += cost
            self.cnt += 1


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # Print stats
        if self.episodesSoFar < self.numTraining:
            if self.params['load_file'] is None:
                log_file = open('./logs/'+'GreedyAgent'+str(self.beta)+'.log','a')
            else:
                log_file = open('./logs/'+ 'Testing_' + self.params['load_file'].split('_')[-1] +'.log','a')

            log_file.write("# %4d | steps: %5d | steps_t: %5d |r: %5d | g: %5d | capsule: %5d | food: %5d | cost: %5d | e: %10f " %
                            (self.episodesSoFar,self.local_cnt, self.cnt, state.data.__dict__['score'],int(state.data.ghost_death), len(state.data.capsules), np.sum(np.sum(state.data.food.__dict__['data'])), float(self.cost_total)/float(self.cnt-self.cnt_last+1), self.params['eps']))
            log_file.write("| won: %r \n" % ((state.data.__dict__['_win'])))
            #sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | r: %5d | g: %5d | capsule: %5d | food: %5d | cost: %5d | e: %10f " %
            #                 (self.episodesSoFar,self.local_cnt, self.cnt, state.data.__dict__['score'], int(state.data.ghost_death), len(state.data.capsules), np.sum(np.sum(state.data.food.__dict__['data'])),float(self.cost_total)/float(self.cnt-self.cnt_last+1), self.params['eps']))
            #sys.stdout.write("| won: %r \n" % ((state.data.__dict__['_win'])))
            #sys.stdout.flush()

            self.cost_total = 0
            self.cnt_last = self.cnt
            self.images = deque(maxlen=self.params['no_frames'])

        if self.episodesSoFar % 5 == 0:
            self.qnet_target.model.set_weights(self.qnet.model.get_weights())

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
