
from tensorflow.python.keras.backend import shape
from DQN_sympathy import *
from state_action_reward_predictor import *
from next_pacman_state_predictor import *
from collections import deque
import random,util,math
from featureExtractors import *
import time
import sys

class Sympathy:
    def __init__(self, numTraining, load_name = None, beta = 1):
        self.featExtractor = util.lookup('SimpleExtractor', globals())()
        self.discount = 0.8
        self.params = {

            #Model saves
            'load_file': None,
            'save_file': 'save',
            'save_interval': 40,

            'training_start': 20, # Number of steps to run before training of DQN starts
            'batch_size': 16, #Replay memory batch size
            'memory_size': 10000, #Size of replay memory
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

            # Update reward prediction
            'reward_update': 50,
            'number_reward_features': 5,

            'window_size_H': 2*7-1,
            'window_size_W': 2*7-1,

            # Selfishness value
            'beta': 0.5
        }

        self.params['beta'] = beta

        if load_name is not None:
            self.params['load_file'] = load_name
            self.params['save_file'] = None
            self.params['training_start'] = 0
            self.params['eps'] = 0

        # Sympathetic DQN
        self.sympathy_qnet = DQNSympathy(self.params)
        self.sympathy_target_qnet = DQNSympathy(self.params)

        # Function to predict reward vector from state and action of Pacman
        self.sa_reward = SA_reward(self.params)
        self.pacman_nextstate = Pacman_nextState(self.params)
        self.ghost_nextstate = Pacman_nextState(self.params)

        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())

        # Place to copy ghost model and pacman model
        self.pacman_model = None
        self.ghost_model = None

        # Ghost IRL rewards
        self.ghost_rewards = None

        # Initialise replay memory
        self.replay_mem = deque()

        # EpisodesSoFar
        self.episodesSoFar = 0

        # NumTraining
        self.numTraining = numTraining

        # Counters
        self.local_cnt = 0
        self.cost_total = 0
        self.cnt = 0
        self.cnt_last = 0
        self.sa_reward_error = 0
        self.pacman_nextstate_error = 0
        self.ghost_nextstate_error = 0

        self.beta_harm_ghost = 0
        self.beta_harm_ghost_cnt = 0
        self.beta_killedbyGhost = 0
        self.beta_killedbyGhost_cnt = 0


    def actionToInt(self,action):
        actionDict = {'North':0,'South':1,'East':2,'West':3,'Stop':4}
        return actionDict[action]

    def registerInitialState(self, state):
        self.startEpisode()

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastPacmanState = None
        self.lastPacmanAction = None
        self.lastGhostState = None
        self.lastGhostAction = None
        self.sympathetic_training_time = 0

    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if (not self.lastPacmanState is None) and (not self.lastGhostState is None):

            # Pacman reward
            rewardPacman = state.getScore() - self.lastPacmanState.getScore()

            rewardFeatExtractor = RewardFeatureExtractor()
            rewardfeatures = np.array(rewardFeatExtractor.getFeatures(self.lastPacmanState, state))

            rewardGhost = rewardfeatures

            self.observeTransition(self.lastPacmanState, self.lastPacmanAction, self.lastGhostState, self.lastGhostAction, state, rewardPacman,rewardGhost)

        return state

    def observeTransition(self, lastPstate,lastPaction,lastGstate,lastGaction,nextState,rewardPacman,rewardGhost):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        t = time.time()
        self.update(lastPstate,lastPaction,lastGstate,lastGaction,nextState,rewardPacman,rewardGhost)
        self.sympathetic_training_time += (time.time()-t)

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
        legalActions = state.getLegalActions()
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

        self.doActionPacman(state,action)
        return action

    def doActionPacman(self,state,action):

        self.lastPacmanState = state.deepCopy()
        self.lastPacmanAction = action
        self.lastGhostAction = None
        self.lastGhostState = None

    def doActionGhost(self,state,action):

        self.lastGhostState = state.deepCopy()
        self.lastGhostAction = action

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = state.getLegalActions() #all the legal actions

        value = self.getValue(state)
        for action in legal_actions:
            if (value == self.getQValue(state, action)):
                return action

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = state.getLegalActions()
        if 'Stop' in legal_actions:
            legal_actions.remove('Stop')

        values = [self.getQValue(state, action) for action in legal_actions]
        if (values):
            return max(values)
        else:
            return 0.0

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        state_features = self.featExtractor.getStateVector_window(state,'P', self.params['window_size_W'],self.params['window_size_H'])

        action_no = self.actionToInt(action)

        state_only = state_features[:-5].reshape(1,self.params['window_size_H'],self.params['window_size_W'],1)
        power_only = state_features[-5].reshape(1,)
        x_self = state_features[-4].reshape(1,)
        y_self = state_features[-3].reshape(1,)
        x_other = state_features[-2].reshape(1,)
        y_other = state_features[-1].reshape(1,)

        QValue = self.sympathy_qnet.model.predict([state_only,power_only,x_self,y_self,x_other,y_other])
        QValue = QValue[0][action_no]

        return QValue

    def update(self, lastPstate,lastPaction,lastGstate,lastGaction,nextState,rewardPacman, rewardGhost):
        """
           Should update your weights based on transition
        """
        self.local_cnt += 1

        if self.local_cnt == 1:
            self.initial_state = lastPstate

        self.last_PacmanState = self.featExtractor.getStateVector_window(lastPstate, 'P', self.params['window_size_W'],self.params['window_size_H'])
        self.last_PacmanAction = self.actionToInt(lastPaction)

        ###########################################################
        # Change rewards
        if rewardPacman > 20:
            self.last_rewardPacman = 50.    # Eat ghost   (Yum! Yum!)
        elif rewardPacman > 0:
            self.last_rewardPacman = 10.    # Eat food    (Yum!)
        elif rewardPacman < -10:
            self.last_rewardPacman = -500.  # Get eaten   (Ouch!) -500
            self.won = False
        elif rewardPacman < 0:
            self.last_rewardPacman = -1.    # Punish time (Pff..)
        elif rewardPacman > 400:
            self.last_rewardPacman = 100.

        ########################################################

        if self.lastGhostState == None:

            self.last_GhostState = self.featExtractor.getStateVector_window(nextState, 'G', self.params['window_size_W'],self.params['window_size_H'])
            self.last_GhostAction = None

        else:

            self.last_GhostState = self.featExtractor.getStateVector_window(lastGstate, 'G', self.params['window_size_W'],self.params['window_size_H'])
            self.last_GhostAction = self.actionToInt(lastGaction)

        self.last_rewardGhost = rewardGhost

        self.current_Pacmanstate = self.featExtractor.getStateVector_window(nextState, 'P', self.params['window_size_W'],self.params['window_size_H'])
        self.current_Ghoststate = self.featExtractor.getStateVector_window(nextState, 'G', self.params['window_size_W'],self.params['window_size_H'])

        #Check whether next state is terminal state
        self.terminal_state = False
        if len(nextState.getLegalActions()) == 0:
            self.terminal_state = True

        # Get reward vector
        rewardFeatExtractor = RewardFeatureExtractor()
        if self.lastGhostState == None:
            reward_vector = np.array(rewardFeatExtractor.getFeatures(lastPstate, nextState))
        else:
            reward_vector = np.array(rewardFeatExtractor.getFeatures(lastPstate, lastGstate))

        #Store experience in memory
        experience = (self.last_PacmanState, float(self.last_rewardPacman), self.last_PacmanAction,self.last_GhostState, self.last_rewardGhost, self.last_GhostAction, self.current_Pacmanstate, self.current_Ghoststate, self.terminal_state,reward_vector,lastPstate, lastGstate, nextState)
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
                layout_name = nextState.data.layout.__dict__['name']
                name = layout_name + '_Sympathetic_beta_' + str(self.params['beta'])
                save_location = 'saves/' + name + "_" + str(self.episodesSoFar)
                self.sympathy_qnet.model.save(save_location)

        # Train model
        if self.episodesSoFar < self.numTraining:
            self.train()


        # Traing state-action reward predictor
        if self.episodesSoFar < self.numTraining:
            self.pacman_nextState_update()
            self.ghost_nextState_update()
            #self.reward_function_update()

    def pacman_nextState_update(self):

        """
        Code for modeling next state as seen by Pacman
        """
        def TDerror(self,batch_s,batch_p_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o,batch_a,batch_r):

            # Calculate error of state action reward function:
            r_vector = self.pacman_nextstate.model.predict([batch_s,batch_p_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o,batch_a])

            prediction = np.zeros(r_vector.shape)
            target = np.zeros(r_vector.shape)

            for r in range(self.params['batch_size']):
                prediction[r] = r_vector[r]
                target[r] = batch_r[r]

            error = np.sum(np.mean(np.abs(target-prediction),axis = 1))

            return error

        if (self.local_cnt > self.params['training_start']):

            # Prepare batch to sample from for training
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = []  # States (s)
            batch_x_s_s = []
            batch_y_s_s = []
            batch_x_o_s = []
            batch_y_o_s = []

            batch_a = []  # Actions (a)
            batch_p_s = [] # Power pellet active current state (s)
            batch_p_n = [] # Power pellet active next state (n)

            batch_target = []

            for i in batch:
                batch_s.append(i[0][:-5].reshape(self.params['window_size_H'],self.params['window_size_W'],1))
                batch_a.append(i[2])
                batch_p_s.append(i[0][-5])
                batch_x_s_s.append(i[0][-4])
                batch_y_s_s.append(i[0][-3])
                batch_x_o_s.append(i[0][-2])
                batch_y_o_s.append(i[0][-1])

                if i[8]:
                    nextState_info = self.featExtractor.getStateVector_window(i[12], 'P', self.params['window_size_W'],self.params['window_size_H'])
                else:
                    nextState_info = self.featExtractor.getStateVector_window(i[11], 'P', self.params['window_size_W'],self.params['window_size_H'])

                batch_p_n.append(nextState_info[-5])
                batch_target.append(nextState_info)

            batch_s = np.array(batch_s)
            batch_a = np.array(batch_a)
            batch_p_s = np.array(batch_p_s)
            batch_x_s_s = np.array(batch_x_s_s)
            batch_y_s_s = np.array(batch_y_s_s)
            batch_x_o_s = np.array(batch_x_o_s)
            batch_y_o_s = np.array(batch_y_o_s)
            batch_p_n = np.array(batch_p_n)
            batch_target = np.array(batch_target)

            self.pacman_nextstate.model.fit([batch_s,batch_p_s,batch_x_s_s,batch_y_s_s,batch_x_o_s,batch_y_o_s,batch_a,batch_p_n], batch_target, shuffle=True, epochs=10, batch_size=len(batch_s),verbose=0)

            error = self.pacman_nextstate.model.history.history['loss'][-1]
            self.pacman_nextstate_error += error

    def ghost_nextState_update(self):

        """
        Code for modeling next state as seen by Ghost
        """
        def TDerror(self,batch_s,batch_p_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o,batch_a,batch_r):

            # Calculate error of state action reward function:
            r_vector = self.ghost_nextstate.model.predict([batch_s,batch_p_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o,batch_a])

            prediction = np.zeros(r_vector.shape)
            target = np.zeros(r_vector.shape)

            for r in range(self.params['batch_size']):
                prediction[r] = r_vector[r]
                target[r] = batch_r[r]

            error = np.sum(np.mean(np.abs(target-prediction),axis = 1))

            return error

        if (self.local_cnt > self.params['training_start']):

            # Prepare batch to sample from for training
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = []  # States (s)
            batch_x_s_s = []
            batch_y_s_s = []
            batch_x_o_s = []
            batch_y_o_s = []

            batch_a = []  # Actions (a)
            batch_p_s = [] # Power pellet active current state (s)
            batch_p_n = [] # Power pellet active next state (n)

            batch_target = []

            for i in batch:

                ghost_state = self.featExtractor.getStateVector_window(i[10], 'G', self.params['window_size_W'],self.params['window_size_H'])

                batch_s.append(ghost_state[:-5].reshape(self.params['window_size_H'],self.params['window_size_W'],1))
                batch_p_s.append(ghost_state[-5])
                batch_x_s_s.append(ghost_state[-4])
                batch_y_s_s.append(ghost_state[-3])
                batch_x_o_s.append(ghost_state[-2])
                batch_y_o_s.append(ghost_state[-1])

                batch_a.append(i[2]) #Pacmans action

                if i[8]:
                    nextState_info = self.featExtractor.getStateVector_window(i[12], 'G', self.params['window_size_W'],self.params['window_size_H'])
                else:
                    nextState_info = self.featExtractor.getStateVector_window(i[11], 'G', self.params['window_size_W'],self.params['window_size_H'])

                batch_p_n.append(nextState_info[-5])

                batch_target.append(nextState_info)

            batch_s = np.array(batch_s)
            batch_a = np.array(batch_a)
            batch_p_s = np.array(batch_p_s)
            batch_x_s_s = np.array(batch_x_s_s)
            batch_y_s_s = np.array(batch_y_s_s)
            batch_x_o_s = np.array(batch_x_o_s)
            batch_y_o_s = np.array(batch_y_o_s)
            batch_p_n = np.array(batch_p_n)
            batch_target = np.array(batch_target)

            self.ghost_nextstate.model.fit([batch_s,batch_p_s,batch_x_s_s,batch_y_s_s,batch_x_o_s,batch_y_o_s,batch_a,batch_p_n], batch_target, shuffle=True, epochs=10, batch_size=len(batch_s),verbose=0)

            error = self.ghost_nextstate.model.history.history['loss'][-1]

            self.ghost_nextstate_error += error

    # Training state-action reward function
    def reward_function_update(self):
        """
        Add code for state-action reward function update
        DO a batch sample from replay memory
        input: pacman state, action
        output: what the actual reward vector was for that state transition (before the Ghost moved)
        """
        def TDerror(self,batch_s,batch_p_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o,batch_a,batch_r):

            # Calculate error of state action reward function:
            r_vector = self.sa_reward.model.predict([batch_s,batch_p_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o,batch_a])

            prediction = np.zeros(r_vector.shape)
            target = np.zeros(r_vector.shape)

            for r in range(self.params['batch_size']):
                prediction[r] = r_vector[r]
                target[r] = batch_r[r]

            error = np.sum(np.mean(np.abs(target-prediction),axis = 1))

            return error

        if (self.local_cnt > self.params['training_start']):

            # Prepare batch to sample from for training
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = []  # Pacman States (s)
            batch_p_s = [] # Power pellet active current state (s)
            batch_a = []  # Actions (a)
            batch_r = []  # Rewards (r) - state to state transition
            batch_x_s = []
            batch_y_s = []
            batch_x_o = []
            batch_y_o = []

            for i in batch:
                batch_s.append(i[0][:-5].reshape(self.params['window_size_H'],self.params['window_size_W'],1))
                batch_r.append(i[9])
                batch_a.append(i[2])
                batch_p_s.append(i[0][-5])
                batch_x_s.append(i[0][-4])
                batch_y_s.append(i[0][-3])
                batch_x_o.append(i[0][-2])
                batch_y_o.append(i[0][-1])

            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = np.array(batch_a)
            batch_p_s = np.array(batch_p_s)
            batch_x_s = np.array(batch_x_s)
            batch_y_s = np.array(batch_y_s)
            batch_x_o = np.array(batch_x_o)
            batch_y_o = np.array(batch_y_o)

            r_vector = batch_r

            self.sa_reward.model.fit([batch_s,batch_p_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o,batch_a], r_vector, shuffle=True, epochs=10, batch_size=len(batch_s),verbose=0)

            error = TDerror(self,batch_s,batch_p_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o,batch_a,batch_r)

            self.sa_reward_error += error

    #Training
    def train(self):

        def TDerror(self,batch_s,batch_x_s_s,batch_y_s_s,batch_x_o_s,batch_y_o_s,batch_a,batch_r,batch_t,batch_p_s,B1_val,B2_val):

            # Calculate TD error:
            q_vals = self.sympathy_qnet.model.predict([batch_s,batch_p_s,batch_x_s_s,batch_y_s_s,batch_x_o_s,batch_y_o_s])

            prediction = np.zeros(self.params['batch_size'],)
            target = np.zeros(self.params['batch_size'],)

            for r in range(self.params['batch_size']):
                prediction[r] = q_vals[r][batch_a[r]]
                Remp =  B1_val[r]*(batch_r[r]) + B2_val[r]*l1_scale*(batch_gr[r])
                if batch_t[r] == False:
                    target[r] = Remp + self.discount*q_emp_next[r]
                else:
                    target[r] = Remp

            error = np.sum(np.abs(target-prediction))

            return error

        def QvalueFunction_setting(numerator_Qg, numerator_Qp, Q_differences):

            c = 1
            x = c*(numerator_Qg-numerator_Qp)/max(Q_differences)

            B1 = 1/(1+np.exp(-1*x))
            B2 = (1-B1)

            return B1, B2

        # Start training if the minimum number of steps have been taken
        if (self.local_cnt > self.params['training_start']):

            if np.sum(np.abs(self.ghost_rewards)) != 0:
                l1_scale = np.sum(np.abs([10,100,50,-500,-1]))/np.sum(np.abs(self.ghost_rewards))
            else:
                l1_scale = 1

            # Prepare batch to sample from for training
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = []  # States (s)
            batch_r = []  # Rewards (r)
            batch_a = []  # Actions (a)
            batch_p_s = [] # Power pellet active current state (s)
            batch_gs = []  # Ghost States (s)
            batch_gr = []  # Ghost Rewards (r)
            batch_ga = []  # Ghost Actions (a)
            batch_g_s = [] # Ghost Power pellet active current state (s)
            batch_pn = []  # Next states (s')
            batch_p_n = [] # Power pellet active next state (n)
            batch_gn = []  # Ghost Next states (s')
            batch_g_n = [] # Ghost Power pellet active next state (n)
            batch_t = []  # Terminal state (t)
            B1_val = []
            B2_val = []

            batch_gx_s_n = []
            batch_gy_s_n = []
            batch_gx_o_n = []
            batch_gy_o_n = []

            batch_x_s_s = []
            batch_y_s_s = []
            batch_x_o_s = []
            batch_y_o_s = []
            batch_x_s_n = []
            batch_y_s_n = []
            batch_x_o_n = []
            batch_y_o_n = []

            for i in batch:
                batch_s.append(i[0][:-5].reshape(self.params['window_size_H'],self.params['window_size_W'],1))
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_p_s.append(i[0][-5])
                batch_x_s_s.append(i[0][-4])
                batch_y_s_s.append(i[0][-3])
                batch_x_o_s.append(i[0][-2])
                batch_y_o_s.append(i[0][-1])

                batch_gs.append(i[3][:-5].reshape(self.params['window_size_H'],self.params['window_size_W'],1))
                batch_gr.append(float(np.dot(self.ghost_rewards,i[4])))

                avail_actions_list = [g for g in i[10].getLegalActions() if g != 'Stop']
                avail_actions = []
                reward_prediction = []
                s = i[0][:-5].reshape(1,self.params['window_size_H'],self.params['window_size_W'],1)
                p_s = i[0][-5].reshape(1,1)
                x_s = i[0][-4].reshape(1,1)
                y_s = i[0][-3].reshape(1,1)
                x_o = i[0][-2].reshape(1,1)
                y_o = i[0][-1].reshape(1,1)

                ghost_position = self.featExtractor.getStateVector_window(i[10], 'G', self.params['window_size_W'],self.params['window_size_H'])
                gs = ghost_position[:-5].reshape(1,self.params['window_size_H'],self.params['window_size_W'],1)
                gp_s = ghost_position[-5].reshape(1,1)
                gx_s = ghost_position[-4].reshape(1,1)
                gy_s = ghost_position[-3].reshape(1,1)
                gx_o = ghost_position[-2].reshape(1,1)
                gy_o = ghost_position[-1].reshape(1,1)

                batch_ga.append(i[5])
                batch_g_s.append(i[3][-1])

                batch_pn.append(i[6][:-5].reshape(self.params['window_size_H'],self.params['window_size_W'],1))
                batch_p_n.append(i[6][-5])
                batch_x_s_n.append(i[6][-4])
                batch_y_s_n.append(i[6][-3])
                batch_x_o_n.append(i[6][-2])
                batch_y_o_n.append(i[6][-1])

                batch_gn.append(i[7][:-5].reshape(self.params['window_size_H'],self.params['window_size_W'],1))
                batch_g_n.append(i[7][-5])
                batch_gx_s_n.append(i[7][-4])
                batch_gy_s_n.append(i[7][-3])
                batch_gx_o_n.append(i[7][-2])
                batch_gy_o_n.append(i[7][-1])

                batch_t.append(i[8])

                if (self.params["beta"] != 0):
                    B1 = 1
                    B2 = 0
                else:

                    if i[8]:

                        # Ghost current
                        pos_ghost_last = 3
                        gs = i[pos_ghost_last][:-5].reshape(1,self.params['window_size_H'],self.params['window_size_W'],1)
                        gp_s = i[pos_ghost_last][-5].reshape(1,1)
                        gx_s = i[pos_ghost_last][-4].reshape(1,1)
                        gy_s = i[pos_ghost_last][-3].reshape(1,1)
                        gx_o = i[pos_ghost_last][-2].reshape(1,1)
                        gy_o = i[pos_ghost_last][-1].reshape(1,1)

                        # Pacman current
                        pos_pacman_last = self.featExtractor.getStateVector_window(i[12], 'P', self.params['window_size_W'],self.params['window_size_H'])
                        ps = pos_pacman_last[:-5].reshape(1,self.params['window_size_H'],self.params['window_size_W'],1)
                        pp_s = pos_pacman_last[-5].reshape(1,1)
                        px_s = pos_pacman_last[-4].reshape(1,1)
                        py_s = pos_pacman_last[-3].reshape(1,1)
                        px_o = pos_pacman_last[-2].reshape(1,1)
                        py_o = pos_pacman_last[-1].reshape(1,1)

                        numerator_Qg = l1_scale*max(self.ghost_model.model.predict([gs,gp_s,gx_s,gy_s,gx_o,gy_o])[0])
                        numerator_Qp = 0

                    elif i[9][2] == 1:

                        # Pacman current
                        pos_pacman_last = self.featExtractor.getStateVector_window(i[11], 'P', self.params['window_size_W'],self.params['window_size_H'])
                        ps = pos_pacman_last[:-5].reshape(1,self.params['window_size_H'],self.params['window_size_W'],1)
                        pp_s = pos_pacman_last[-5].reshape(1,1)
                        px_s = pos_pacman_last[-4].reshape(1,1)
                        py_s = pos_pacman_last[-3].reshape(1,1)
                        px_o = pos_pacman_last[-2].reshape(1,1)
                        py_o = pos_pacman_last[-1].reshape(1,1)

                        numerator_Qg = 0
                        numerator_Qp = max(self.pacman_model.model.predict([ps,pp_s,px_s,py_s,px_o,py_o])[0])

                    else:

                        # Ghost current
                        pos_ghost_last = 3
                        gs = i[pos_ghost_last][:-5].reshape(1,self.params['window_size_H'],self.params['window_size_W'],1)
                        gp_s = i[pos_ghost_last][-5].reshape(1,1)
                        gx_s = i[pos_ghost_last][-4].reshape(1,1)
                        gy_s = i[pos_ghost_last][-3].reshape(1,1)
                        gx_o = i[pos_ghost_last][-2].reshape(1,1)
                        gy_o = i[pos_ghost_last][-1].reshape(1,1)

                        # Pacman current
                        pos_pacman_last = self.featExtractor.getStateVector_window(i[11], 'P', self.params['window_size_W'],self.params['window_size_H'])
                        ps = pos_pacman_last[:-5].reshape(1,self.params['window_size_H'],self.params['window_size_W'],1)
                        pp_s = pos_pacman_last[-5].reshape(1,1)
                        px_s = pos_pacman_last[-4].reshape(1,1)
                        py_s = pos_pacman_last[-3].reshape(1,1)
                        px_o = pos_pacman_last[-2].reshape(1,1)
                        py_o = pos_pacman_last[-1].reshape(1,1)

                        numerator_Qg = l1_scale*max(self.ghost_model.model.predict([gs,gp_s,gx_s,gy_s,gx_o,gy_o])[0])
                        numerator_Qp = max(self.pacman_model.model.predict([ps,pp_s,px_s,py_s,px_o,py_o])[0])

                    for a_int in avail_actions_list:
                        avail_actions.append(self.actionToInt(a_int))

                        action_predict = np.array(self.actionToInt(a_int)).reshape(1,1)
                        power_next = np.array(i[6][-5]).reshape(1,1)
                        s_next_pred_model = self.pacman_nextstate.model.predict([s,p_s,x_s,y_s,x_o,y_o,action_predict,power_next])[0]
                        s_next_ghost_pred_model = self.ghost_nextstate.model.predict([gs,gp_s,gx_s,gy_s,gx_o,gy_o,action_predict,power_next])[0]

                        # Predicting denominator based on action
                        pos_pacman_last = s_next_pred_model
                        ps = pos_pacman_last[:-5].reshape(1,self.params['window_size_H'],self.params['window_size_W'],1)
                        pp_s = pos_pacman_last[-5].reshape(1,1)
                        px_s = pos_pacman_last[-4].reshape(1,1)
                        py_s = pos_pacman_last[-3].reshape(1,1)
                        px_o = pos_pacman_last[-2].reshape(1,1)
                        py_o = pos_pacman_last[-1].reshape(1,1)

                        # Ghost current
                        pos_ghost_last = s_next_ghost_pred_model
                        gs = pos_ghost_last[:-5].reshape(1,self.params['window_size_H'],self.params['window_size_W'],1)
                        gp_s = pos_ghost_last[-5].reshape(1,1)
                        gx_s = pos_ghost_last[-4].reshape(1,1)
                        gy_s = pos_ghost_last[-3].reshape(1,1)
                        gx_o = pos_ghost_last[-2].reshape(1,1)
                        gy_o = pos_ghost_last[-1].reshape(1,1)

                        denominator_Qg = l1_scale*max(self.ghost_model.model.predict([gs,gp_s,gx_s,gy_s,gx_o,gy_o])[0])
                        denominator_Qp = max(self.pacman_model.model.predict([ps,pp_s,px_s,py_s,px_o,py_o])[0])

                        reward_prediction.append(np.abs(denominator_Qg-denominator_Qp))

                    B1, B2 = QvalueFunction_setting(numerator_Qg, numerator_Qp, reward_prediction)

                    if (i[8] == 1) and (i[1] < -400):
                        self.beta_killedbyGhost += B1
                        self.beta_killedbyGhost_cnt += 1
                    if i[9][2] == 1:
                        self.beta_harm_ghost += B1
                        self.beta_harm_ghost_cnt += 1

                B1_val.append(B1)
                B2_val.append(B2)

            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = np.array(batch_a)
            batch_p_s = np.array(batch_p_s)
            batch_x_s_s = np.array(batch_x_s_s)
            batch_y_s_s = np.array(batch_y_s_s)
            batch_x_o_s = np.array(batch_x_o_s)
            batch_y_o_s = np.array(batch_y_o_s)

            batch_gs = np.array(batch_gs)
            batch_gr = np.array(batch_gr)
            batch_ga = np.array(batch_ga)
            batch_g_s = np.array(batch_g_s)

            batch_pn = np.array(batch_pn)
            batch_p_n = np.array(batch_p_n)
            batch_x_s_n = np.array(batch_x_s_n)
            batch_y_s_n = np.array(batch_y_s_n)
            batch_x_o_n = np.array(batch_x_o_n)
            batch_y_o_n = np.array(batch_y_o_n)

            batch_gn = np.array(batch_gn)
            batch_g_n = np.array(batch_g_n)
            batch_gx_s_n = np.array(batch_gx_s_n)
            batch_gy_s_n = np.array(batch_gy_s_n)
            batch_gx_o_n = np.array(batch_gx_o_n)
            batch_gy_o_n = np.array(batch_gy_o_n)

            batch_t = np.array(batch_t)

            # Predicted Qvalues and NextQvalues
            q_emp_vals = self.sympathy_qnet.model.predict([batch_s,batch_p_s,batch_x_s_s,batch_y_s_s,batch_x_o_s,batch_y_o_s])
            q_emp_next = self.sympathy_target_qnet.model.predict([batch_pn,batch_p_n,batch_x_s_n,batch_y_s_n,batch_x_o_n,batch_y_o_n])
            q_emp_next = np.max(q_emp_next,axis=1)

            for i in range(len(batch_a)):
                Remp = B1_val[i]*(batch_r[i]) + B2_val[i]*l1_scale*(batch_gr[i])
                if batch_t[i] == False:
                    q_emp_vals[i][batch_a[i]] = Remp + self.discount*q_emp_next[i]
                else:
                    q_emp_vals[i][batch_a[i]] = Remp
                    q_emp_vals = np.concatenate((q_emp_vals, np.zeros((1,4))),axis = 0)
                    batch_s = np.concatenate((batch_s, batch_pn[i].reshape(1,self.params['window_size_H'],self.params['window_size_W'],1)), axis = 0)
                    batch_p_s = np.concatenate((batch_p_s,np.array([batch_p_n[i]])),axis = 0)
                    batch_x_s_s = np.concatenate((batch_x_s_s,np.array([batch_x_s_n[i]])),axis = 0)
                    batch_y_s_s = np.concatenate((batch_y_s_s,np.array([batch_y_s_n[i]])),axis = 0)
                    batch_x_o_s = np.concatenate((batch_x_o_s,np.array([batch_x_o_n[i]])),axis = 0)
                    batch_y_o_s = np.concatenate((batch_y_o_s,np.array([batch_y_o_n[i]])),axis = 0)

            self.sympathy_qnet.model.fit([batch_s,batch_p_s,batch_x_s_s,batch_y_s_s,batch_x_o_s,batch_y_o_s], q_emp_vals, shuffle=True, epochs=10, batch_size=len(batch_s),verbose=0)

            error = TDerror(self,batch_s,batch_x_s_s,batch_y_s_s,batch_x_o_s,batch_y_o_s,batch_a,batch_r,batch_t,batch_p_s,B1_val,B2_val)

            cost = error
            self.cost_total += cost
            self.cnt += 1

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        self.episodesSoFar += 1

        rewardPacman = state.getScore() - self.lastPacmanState.getScore()
        rewardFeatExtractor = RewardFeatureExtractor()
        rewardfeatures = np.array(rewardFeatExtractor.getFeatures(self.lastPacmanState, state))
        rewardGhost = rewardfeatures
        self.observeTransition(self.lastPacmanState, self.lastPacmanAction, self.lastGhostState, self.lastGhostAction, state, rewardPacman,rewardGhost)

        if self.beta_harm_ghost_cnt == 0:
            self.beta_harm_ghost_cnt = 1

        if self.beta_killedbyGhost_cnt == 0:
            self.beta_killedbyGhost_cnt = 1

        # Print stats
        if self.params['load_file'] is None:
            log_file = open('./logs/'+'Sympathetic_beta'+ str(self.params['beta'])+'.log','a')
        else:
            log_file = open('./logs/'+ 'Sympathetic_beta' + str(self.params['beta']) + '_Testing_' + self.params['load_file'].split('_')[-1] +'.log','a')

        log_file.write("# %4d | steps: %5d | steps_t: %5d |r: %5d | g: %5d | capsule: %5d | food: %5d | cost: %5f | p_next_error: %5f | g_next_error: %5f | e: %10f " %
                         (self.episodesSoFar,self.local_cnt, self.cnt, state.data.__dict__['score'],int(state.data.ghost_death), len(state.data.capsules), np.sum(np.sum(state.data.food.__dict__['data'])), float(self.cost_total)/float(self.cnt-self.cnt_last+1), float(self.pacman_nextstate_error)/float(self.cnt-self.cnt_last+1), float(self.ghost_nextstate_error)/float(self.cnt-self.cnt_last+1),self.params['eps']))
        log_file.write("| won: %r \n" % ((state.data.__dict__['_win'])))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | r: %5d | g: %5d | capsule: %5d | food: %5d | cost: %5f | p_next_error: %5f | g_next_error: %5f | | e: %10f " %
                         (self.episodesSoFar,self.local_cnt, self.cnt, state.data.__dict__['score'], int(state.data.ghost_death), len(state.data.capsules), np.sum(np.sum(state.data.food.__dict__['data'])),float(self.cost_total)/float(self.cnt-self.cnt_last+1), float(self.pacman_nextstate_error)/float(self.cnt-self.cnt_last+1), float(self.ghost_nextstate_error)/float(self.cnt-self.cnt_last+1), self.params['eps']))
        sys.stdout.write("| won: %r \n" % ((state.data.__dict__['_win'])))
        sys.stdout.flush()

        if self.params['load_file'] is None:
            log_file = open('./logs/'+'beta_values'+'.log','a')
        log_file.write("# %4d | beta_harm_ghost: %r | beta_killedbyGhost: %r \n" %
                                    (self.episodesSoFar,float(self.beta_harm_ghost)/float(self.beta_harm_ghost_cnt),float(self.beta_killedbyGhost)/float(self.beta_killedbyGhost_cnt)))

        self.cost_total = 0
        self.pacman_nextstate_error = 0
        self.ghost_nextstate_error = 0
        self.cnt_last = self.cnt

        self.beta_harm_ghost = 0
        self.beta_harm_ghost_cnt = 0
        self.beta_killedbyGhost = 0
        self.beta_killedbyGhost_cnt = 0

        if self.episodesSoFar % 5 == 0:
            self.sympathy_target_qnet.model.set_weights(self.sympathy_qnet.model.get_weights())

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass