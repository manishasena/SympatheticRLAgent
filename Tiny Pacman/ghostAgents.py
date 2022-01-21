# ghostAgents.py
# --------------
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

from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util
from collections import deque
import itertools

#from learningAgents import ReinforcementAgent
import time
from featureExtractors import *
from DQN_ghost import *

class GhostAgent( Agent ):
    def __init__( self, index ):
        self.index = index


    def getAction( self, state ):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            action = Directions.STOP
        else:
            action = util.chooseFromDistribution( dist )
        self.doAction(state,action)
        return action

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()



class RandomGhost( GhostAgent ):
    "A ghost that chooses a legal action uniformly at random."
    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist

class DirectionalGhost( GhostAgent ):
    "A ghost that prefers to rush Pacman, or flee when scared."
    def __init__( self, index, numTraining, prob_attack=0.8, prob_scaredFlee=0.8 ):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

        self.numTraining = numTraining

        self.featExtractor = util.lookup('SimpleExtractor', globals())()

        self.discount = 0.8

        self.params = {

            #Model saves
            'load_file': None,
            'save_file': 'save',
            'save_interval': 40,

            'training_start': 100, # Number of steps to run before training of DQN starts
            'batch_size': 16, #Replay memory batch size
            'memory_size': 10000, #Size of replay memory
            'IRL_mem_size': 10000, #reduced to 10,000 because of the stochastic updates
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
            'reward_update': 10,

            'window_size_H': 2*7-1,
            'window_size_W': 2*7-1
        }

        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())

        # EpisodesSoFar
        self.episodesSoFar = 0

        # Ghost DQN
        self.ghost_qnet = DQNGhost(self.params)

        # Initialise replay memory
        self.replay_mem_ghost = deque()
        self.IRL_mem = deque()

        # IRL reward
        self.reward_ghost = np.zeros((1,5)) #np.array([0,2,-1,3,-3]).reshape(1,5) #

        # Testing set of ghost movements
        self.ghost_testset_s = []
        self.ghost_testset_a = []
        self.ghost_testset_ps = []
        self.ghost_testset_xs = []
        self.ghost_testset_ys = []
        self.ghost_testset_xo = []
        self.ghost_testset_yo = []

        self.ghost_cost_total = 0
        self.cnt_ghost = 0
        self.cnt_last_ghost = 0
        self.IRL_error = 0


    def getGhostReward(self):

        return self.reward_ghost

    def getGhostValueFunction(self):

        return self.ghost_qnet

    def getDistribution( self, state ):
        # Read variables from state
        ghostState = state.getGhostState( self.index )
        legalActions = state.getLegalActions( self.index )
        pos = state.getGhostPosition( self.index )
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
        newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
        if isScared:
            bestScore = max( distancesToPacman )
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min( distancesToPacman )
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]
        bestActions = [bestActions[0]]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
        dist.normalize()
        return dist

    def actionToInt(self,action):

        actionDict = {'North':0,'South':1,'East':2,'West':3,'Stop':4}
        return actionDict[action]

    def doAction(self,state,action):

        self.lastStateGhost = state.deepCopy()
        self.lastActionGhost = action

    def registerInitialState(self, state):
        self.startEpisode()

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastStateGhost = None
        self.lastActionGhost = None
        self.ghost_training_time = 0
        self.IRL_mem_time = 0

    def observationFunctionGhost(self, state):
        """
            Observe transition of the ghost agent and update
        """
        #if Ghost_killed == False:
        if not self.lastStateGhost is None:
            self.observeGhostTransition(self.lastStateGhost, self.lastActionGhost, state)
        return state

    def observeGhostTransition(self, state, action, nextState):

        t = time.time()
        self.update(state,action,nextState)
        self.ghost_training_time += (time.time()-t)

    def update(self, state, action, nextState):

        if self.episodesSoFar < self.numTraining:

            # Store ghost moves to use for testing performance of ghost model
            if (len(self.ghost_testset_s) < self.params['training_start']):

                # Memories for classifier training
                ghost_state = self.featExtractor.getStateVector_window(state, 'G', self.params['window_size_W'],self.params['window_size_H'])
                ghost_next_state = self.featExtractor.getStateVector_window(nextState, 'G', self.params['window_size_W'],self.params['window_size_H'])

                experience = (ghost_state, self.actionToInt(action), ghost_next_state)

                exp_s = (experience[0][:-5].reshape(self.params['window_size_H'],self.params['window_size_W'],1))
                exp_a = experience[1]
                exp_p_s = experience[0][-5]
                exp_x_s = experience[0][-4]
                exp_y_s = experience[0][-3]
                exp_x_o = experience[0][-2]
                exp_y_o = experience[0][-1]

                self.ghost_testset_s.append(exp_s)
                self.ghost_testset_a.append(exp_a)
                self.ghost_testset_ps.append(exp_p_s)
                self.ghost_testset_xs.append(exp_x_s)
                self.ghost_testset_ys.append(exp_y_s)
                self.ghost_testset_xo.append(exp_x_o)
                self.ghost_testset_yo.append(exp_y_o)

            if (self.beta == 0):
                # Memories for classifier training
                rewardFeatExtractor = RewardFeatureExtractor()
                rewardfeatures = np.array(rewardFeatExtractor.getFeatures(state, nextState))

                ghost_state = self.featExtractor.getStateVector_window(state, 'G', self.params['window_size_W'],self.params['window_size_H'])
                ghost_next_state = self.featExtractor.getStateVector_window(nextState, 'G', self.params['window_size_W'],self.params['window_size_H'])
                experience = (ghost_state, self.actionToInt(action), ghost_next_state,state,nextState,rewardfeatures)
                self.replay_mem_ghost.append(experience)

                # Memory for IRL
                t = time.time()
                if len(self.IRL_mem) == 0:
                    self.IRL_mem.append(experience)

                    # Loop through and also attach other possible moves and their rewards
                    legalActions = state.getLegalActions( self.index )
                    for action in legalActions:
                        action_indx = self.actionToInt(action)
                        if action_indx != experience[1]:
                            nextState_pred = state.generateSuccessor( 1, action )
                            s2 = self.featExtractor.getStateVector_window(nextState_pred, 'G', self.params['window_size_W'],self.params['window_size_H'])

                            rewardFeatExtractor = RewardFeatureExtractor()
                            rewardfeatures = np.array(rewardFeatExtractor.getFeatures(state, nextState_pred))

                            experience_pred = (ghost_state, self.actionToInt(action), s2,state,nextState_pred,rewardfeatures)
                            self.IRL_mem.append(experience_pred)
                else:
                    inside = False
                    for a in range(len(self.IRL_mem)):
                        if ((experience[0] == self.IRL_mem[a][0]).all()) and (experience[1] == self.IRL_mem[a][1]):
                                inside = True
                        if inside == True:
                            break

                    if inside == False:
                        self.IRL_mem.append(experience)

                        # Loop through and also attach other possible moves and their rewards
                        legalActions = state.getLegalActions( self.index )
                        for action in legalActions:
                            action_indx = self.actionToInt(action)
                            if action_indx != experience[1]:
                                nextState_pred = state.generateSuccessor( 1, action )
                                s2 = self.featExtractor.getStateVector_window(nextState_pred, 'G', self.params['window_size_W'],self.params['window_size_H'])

                                rewardFeatExtractor = RewardFeatureExtractor()
                                rewardfeatures = np.array(rewardFeatExtractor.getFeatures(state, nextState_pred))

                                experience_pred = (ghost_state, self.actionToInt(action), s2,state,nextState_pred,rewardfeatures)
                                self.IRL_mem.append(experience_pred)

                self.IRL_mem_time += (time.time()-t)

                # Remove oldest experience if replay memory full
                if len(self.replay_mem_ghost) > self.params['memory_size']:
                    self.replay_mem_ghost.popleft()

                # Added length limit on IRL mem
                while len(self.IRL_mem) > self.params['IRL_mem_size']:
                    self.IRL_mem.popleft()

                # Update ghost model
                if (len(self.replay_mem_ghost)>0):
                    self.train_ghost_model()

                # Update ghost reward
                if (len(self.replay_mem_ghost)>0):
                    self.reward_prediction()


    # Update ghost model
    def train_ghost_model(self):

        def TDerror_binary(self,batch_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o,batch_a,batch_p_s):

            # Calculate TD error:
            q_vals = self.ghost_qnet.model.predict([batch_s,batch_p_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o])
            perc_correct = np.sum(np.argmax(q_vals,1) == batch_a)/len(batch_a)
            J = 0
            for r in range(len(batch_a)):
                J += -1*np.log(q_vals[r][batch_a[r]])

            error = J/len(batch_a)
            return error, perc_correct

        if (len(self.replay_mem_ghost) > self.params['training_start']):

            batch_size = self.params['batch_size']

            batch = random.sample(self.replay_mem_ghost, batch_size)
            batch_s = []  # States (s)
            batch_a = []  # Actions (a)
            batch_p_s = [] # Power pellet active current state (s)
            batch_state = []
            batch_x_s = []
            batch_y_s = []
            batch_x_o = []
            batch_y_o = []

            for i in batch:
                batch_s.append(i[0][:-5].reshape(self.params['window_size_H'],self.params['window_size_W'],1))
                batch_a.append(i[1])
                batch_p_s.append(i[0][-5])
                batch_x_s.append(i[0][-4])
                batch_y_s.append(i[0][-3])
                batch_x_o.append(i[0][-2])
                batch_y_o.append(i[0][-1])
                batch_state.append(i[3])

            batch_s = np.array(batch_s)
            batch_a = np.array(batch_a)
            batch_p_s = np.array(batch_p_s)
            batch_x_s = np.array(batch_x_s)
            batch_y_s = np.array(batch_y_s)
            batch_x_o = np.array(batch_x_o)
            batch_y_o = np.array(batch_y_o)

            # Convert actions to onehot encoding
            q_vals = np.zeros((batch_size,4))

            pred_q_vals = self.ghost_qnet.model.predict([batch_s,batch_p_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o])

            for i in range(len(batch_a)):
                action_list = batch_state[i].getLegalActions( self.index )
                actions = []
                for act in action_list:
                    actions.append(self.actionToInt(act))
                for r in range(4):
                    if r in actions:
                        if (np.argmax(pred_q_vals[i]) == batch_a[i]) and (r==batch_a[i]):
                            q_vals[i][r] = pred_q_vals[i][r]
                        elif (r == batch_a[i]):
                            q_vals[i][batch_a[i]] = 1
                    else:
                        q_vals[i][r] = pred_q_vals[i][r]


            error, perc_correct = TDerror_binary(self,batch_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o,batch_a,batch_p_s)
            #error, perc_correct = TDerror_binary(self,np.array(self.ghost_testset_s),np.array(self.ghost_testset_a),np.array(self.ghost_testset_ps))
            self.ghost_qnet.model.fit([batch_s,batch_p_s,batch_x_s,batch_y_s,batch_x_o,batch_y_o], q_vals, shuffle=True, epochs=10, batch_size=len(batch_s),verbose=0)

            self.ghost_cost_total += perc_correct
            self.cnt_ghost += 1

    # Predict reward vector of ghost
    def reward_prediction(self):

        if (len(self.replay_mem_ghost) > self.params['training_start']):

            batch_size = self.params['batch_size']
            batch = random.sample(self.IRL_mem, batch_size)

            R_target = []
            rewardfeaturelist = []

            for i in range(self.params['batch_size']):

                s1 = batch[i][0]
                state = batch[i][3]
                legalActions = state.getLegalActions( self.index )

                window = s1[:-5].reshape(1,self.params['window_size_H'],self.params['window_size_W'],1)
                power = s1[-5].reshape(1,1)
                x_self = s1[-4].reshape(1,1)
                y_self = s1[-3].reshape(1,1)
                x_other = s1[-2].reshape(1,1)
                y_other = s1[-1].reshape(1,1)
                rewardfeatures = batch[i][5]
                #state = self.replay_mem_ghost[i][3]

                s2 = batch[i][2]
                nextState = batch[i][4]

                pred_action = batch[i][1]

                """
                for action in legalActions:

                    action_indx = self.actionToInt(action)
                    if action_indx == batch[i][1]:

                        s2 = batch[i][2]
                        nextState = batch[i][4]

                    else:

                        nextState = state.generateSuccessor( 1, action )
                        s2 = self.featExtractor.getStateVector_window(nextState, 'G', self.params['window_size_W'],self.params['window_size_H'])

                    pred_action = action_indx
                """
                window_next = s2[:-5].reshape(1,self.params['window_size_H'],self.params['window_size_W'],1)
                power_next = s2[-5].reshape(1,1)
                x_self_next = s2[-4].reshape(1,1)
                y_self_next = s2[-3].reshape(1,1)
                x_other_next = s2[-2].reshape(1,1)
                y_other_next = s2[-1].reshape(1,1)
                #nextState = s2

                q = self.ghost_qnet.model_bsm.predict([window,power,x_self,y_self,x_other,y_other])
                q = q[0][pred_action]

                #rewardFeatExtractor = RewardFeatureExtractor()
                #rewardfeatures = np.array(rewardFeatExtractor.getFeatures(state, nextState))

                if rewardfeatures[2] == 1:

                    q_max = 0.0

                else:
                    q_max = max(self.ghost_qnet.model_bsm.predict([window_next,power_next,x_self_next,y_self_next,x_other_next,y_other_next])[0])

                R = q - self.discount*q_max

                R_target.append(R)
                rewardfeaturelist.append(rewardfeatures)

                pred_R = np.dot(self.reward_ghost, rewardfeatures)
                difference_R = (pred_R - R)
                self.reward_ghost -= 0.2 * rewardfeatures * difference_R

            # Rough estimate of reward error
            error = 0
            for i in range(self.params['batch_size']):
                error +=  np.sum(np.abs(np.dot(self.reward_ghost, rewardfeaturelist[i]) - R_target[i]))

            self.IRL_error += error


    def final(self, state,pacman_terminal = True):

        if not self.lastStateGhost is None:
            self.observeGhostTransition(self.lastStateGhost, self.lastActionGhost, state)

        if pacman_terminal:

            self.episodesSoFar += 1

            if self.episodesSoFar < self.numTraining:

                avg_perc = float(self.ghost_cost_total)/float(self.cnt_ghost-self.cnt_last_ghost+1)
                avg_IRLerror = float(self.IRL_error)/float(self.cnt_ghost-self.cnt_last_ghost+1)
                self.ghost_cost_total = 0
                self.IRL_error = 0
                self.cnt_last_ghost = self.cnt_ghost

                """
                # IRL on rewards
                # Update predicted reward value of ghost
                if self.beta != 1:
                    if (self.episodesSoFar % self.params['reward_update'] == 1) and (len(self.IRL_mem)>0):
                        t = time.time()
                        self.reward_prediction()
                        print("IRL time:" + str(time.time()-t))
                """

                #print("Ghost training time:" + str(self.ghost_training_time))
                print("IRL mem filling time:" + str(self.IRL_mem_time))

                log_file = open('./logs/'+'Ghost_Model_accuracy'+ str(self.beta)+'.log','a')
                log_file.write("# %4d | Accuracy: %10f \n" % ((self.episodesSoFar, avg_perc)))

                log_file = open('./logs/'+'IRL_accuracy'+ str(self.beta)+'.log','a')
                log_file.write("# %4d | Accuracy: %10f \n" % ((self.episodesSoFar, avg_IRLerror)))

                # Print stats
                #if self.episodesSoFar % self.params['reward_update'] == 1:
                if self.params['load_file'] is None:
                    log_file = open('./logs/'+'reward' + str(self.beta) + '.log','a')
                else:
                    log_file = open('./logs/'+ 'Testing_Reward' + self.params['load_file'].split('_')[-1] +'.log','a')

                log_file.write("# %4d | reward: %r \n" %
                                (self.episodesSoFar,list(self.reward_ghost[0])))



