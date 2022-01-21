# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util
import numpy as np

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

    def getStateVector_window(self,state,agent='P', window_size_W = 5, window_size_H = 5):

        window_size_W = (window_size_W+1)/2
        window_size_H = (window_size_H+1)/2

        # Get positions of food and walls
        food = state.getFood()
        walls = state.getWalls()

        if agent == 'P':
            x, y = state.getPacmanPosition()
            x_other, y_other = state.getGhostPositions()[0]
        else:
            x, y = state.getGhostPositions()[0]
            x_other, y_other = state.getPacmanPosition()

        # For each square around agent, check presence of:
        win_H = int(2*window_size_H-1)
        win_W = int(2*window_size_W-1)
        window = np.zeros((win_H,win_W))

        if agent == 'P':
            window[int(win_H/2),int(win_W/2)] = 0
        else:
            window[int(win_H/2),int(win_W/2)] = 0

        x_c = int(win_H/2)
        y_c = int(win_W/2)
        for i in np.arange(-1*int(window_size_H/2),int(window_size_H/2)+0.5,0.5):
            for j in np.arange(-1*int(window_size_W/2),int(window_size_W/2)+0.5,0.5):
                if ((i != 0) or (j != 0)):
                    x_i = x+i
                    y_j = y+j

                    x_c_i = int(x_c + 2*i)
                    y_c_i = int(y_c + 2*j)

                    if (x_i < 0) or (y_j < 0) or (x_i >= walls.width) or (y_j >= walls.height):

                        #window[x_c_i][y_c_i] = 2
                        pass

                    else:
                        # Check if there is a wall in the cell
                        if (x_i % 1 == 0) and (y_j % 1 == 0):
                            x_i = int(x_i)
                            y_j = int(y_j)
                            if walls[x_i][y_j]:
                                window[x_c_i][y_c_i] = 1/4

                            # Check if there is food in the cell
                            if food[x_i][y_j]:
                                window[x_c_i][y_c_i] = 3/4

                            # Check if there is pellet in the cell
                            if (x_i,y_j) in state.getCapsules():
                                window[x_c_i][y_c_i] = 2/4

                        # Check if other agent in the cell
                        if agent == 'P':
                            if (x_i,y_j) in state.getGhostPositions():
                                window[x_c_i][y_c_i] = 4/4
                        else:
                            if (x_i,y_j) in [state.getPacmanPosition()]:
                                window[x_c_i][y_c_i] = 4/4

        features = window.ravel()

        # Power pellet active
        if state.data.agentStates[1].scaredTimer > 0:
            features = np.concatenate((features,np.array([1])))
        else:
            features = np.concatenate((features,np.array([0])))

        # x and y positions of both agents
        features = np.concatenate((features,np.array([x/(window.shape[0]-1),y/(window.shape[1]-1),x_other/(window.shape[0]-1),y_other/(window.shape[1]-1)])))

        # Number of food left
        #features = np.concatenate((features,np.array([state.getNumFood()])))

        return features

    def getStateVector(self,state,agent='P'):

        # Get positions of food and walls
        food = state.getFood()
        walls = state.getWalls()

        # First prepare feature vector for the 3 global parameters
        features = np.zeros([4,])

        if agent == 'P':
            features[0] = 1
            x, y = state.getPacmanPosition()
        else:
            x, y = state.getGhostPositions()

        # Check if pellet is active (by checking whether the ghost is scared)
        if state.data.agentStates[1].scaredTimer > 0:
            features[1] = 1

        # Check if all food has been eaten
        if (state.getNumFood() == 1):
            if closestFood((x,y), food, walls) == 1:
                features[2] = 1

        # Distance to closest food
        dist = closestFood((x, y), food, walls)
        if dist is not None:
            features[3] = float(dist) / (walls.width * walls.height)


        # For each square around agent, check presence of:
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                if ((i != 0) or (j != 0)):
                    x_i = x+i
                    y_j = y+j
                    tmp = np.zeros([4,])

                    # Check if there is a wall in the cell
                    if walls[x_i][y_j]:
                        tmp[0] = 1

                    # Check if there is food in the cell
                    if food[x_i][y_j]:
                        tmp[1] = 1

                    # Check if there is pellet in the cell
                    if (x_i,y_j) in state.getCapsules():
                        tmp[2] = 1

                    # Check if other agent in the cell
                    if agent == 'P':
                        if (x_i,y_j) in state.getGhostPositions():
                            tmp[3] = 1
                    else:
                        if (x_i,y_j) in state.getPacmanPosition():
                            tmp[3] = 1

                    # concatenate tmp to features
                    features = np.concatenate((features,tmp))
        return features

    def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """
        def getWallMatrix(state,observation):
            """ Return matrix with wall coordinates set to 1 """
            grid = state.data.layout.walls

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    if grid[j][i]:
                        observation[0][-1-i][j] = 0
                        observation[1][-1-i][j] = 0
                        observation[2][-1-i][j] = 1

            return observation

        def getPacmanMatrix(state,observation):
            """ Return matrix with pacman coordinates set to 1 """

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    observation[0][-1-int(pos[1])][int(pos[0])] = 1
                    observation[1][-1-int(pos[1])][int(pos[0])] = 1
                    observation[2][-1-int(pos[1])][int(pos[0])] = 1

            return observation

        def getGhostMatrix(state,observation):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            #matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        observation[0][-1-int(pos[1])][int(pos[0])] = 0
                        observation[1][-1-int(pos[1])][int(pos[0])] = 1
                        observation[2][-1-int(pos[1])][int(pos[0])] = 0

            return observation

        def getScaredGhostMatrix(state,observation):
            """ Return matrix with ghost coordinates set to 1 """

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        observation[0][-1-int(pos[1])][int(pos[0])] = 1
                        observation[1][-1-int(pos[1])][int(pos[0])] = 0
                        observation[2][-1-int(pos[1])][int(pos[0])] = 0.5

            return observation

        def getFoodMatrix(state,observation):
            """ Return matrix with food coordinates set to 1 """
            grid = state.data.food

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    if grid[j][i]:
                        observation[0][-1-i][j] = 0.4
                        observation[1][-1-i][j] = 0.2
                        observation[2][-1-i][j] = 0

            return observation

        def getCapsulesMatrix(state,observation):
            """ Return matrix with capsule coordinates set to 1 """
            capsules = state.data.layout.capsules

            if len(state.getCapsules()) > 0:

                for i in capsules:
                    # Insert capsule cells vertically reversed into matrix
                    observation[0][-1-i[1], i[0]] = 1
                    observation[1][-1-i[1], i[0]] = 0
                    observation[2][-1-i[1], i[0]] = 0

            return observation

        # Helpful preprocessing taken from github.com/ageron/tiny-dqn
        def process_frame(img):
            img = np.dot(img, np.array([0.30,0.59,0.11]))
            #mspacman_color = np.array([210, 164, 74]).mean()
            #img = frame[1:176:2, ::2]    # Crop and downsize
            #img = img.mean(axis=2)       # Convert to greyscale
            #img[img==mspacman_color] = 0 # Improve contrast by making pacman white
            #img = (img - 0.5) / 0.5 - 1  # Normalize from -1 to 1.

            return img

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        width, height = state.data.layout.width, state.data.layout.height
        #width, height = self.params['width'], self.params['height']

        observation = np.zeros((3, height, width))
        observation = getWallMatrix(state,observation)
        observation = getPacmanMatrix(state,observation)
        observation = getGhostMatrix(state,observation)
        observation = getScaredGhostMatrix(state,observation)
        observation = getFoodMatrix(state,observation)
        observation = getCapsulesMatrix(state,observation)

        observation = np.swapaxes(observation, 0, 2)

        observation = process_frame(observation)
        observation = observation.reshape(9,8,1)

        return observation

class RewardFeatureExtractor:
    """
    Returns features of (s,a,s') that are to be used as features for the reward function
    - no bias term
    - whether food has been eaten
    - whether pacman has won
    - number of scared ghosts eaten
    - whether collide wth unscared ghost, i.e. whether pacman has lost
    [Not including the spurious features below initially]
    - how far away the next food is
    - distance of the closest scared ghost
    - distance of the closest unscared ghost
    """

    def getFeatures(self, state, new_state):
        num_feat = 5
        food_ind = 0
        won_ind = 1
        ghostsEaten_ind = 2
        lost_ind = 3
        time_step = 4
        # Not including the spurious features right now

        features = [0.0]*num_feat

        features[time_step] = 1

        features[food_ind] = state.getNumFood() - new_state.getNumFood() # 0    # whether food was eaten in this step

        if features[food_ind] != 0:
            features[time_step] = 0

        if new_state.data._win:
            features[won_ind] = 1.0
            features[food_ind] = 0 # Set food to 0 if pwin active
            features[time_step] = 0
        #if (new_state.getPacmanPosition() in new_state.getGhostPositions()) and (new_state.data.agentStates[1].scaredTimer>0):
        #    features[won_ind] = 1

        for ind in range(1, len(new_state.data._eaten)):
            #if not state.data._eaten[ind]:
            if new_state.data._eaten[ind]:
                features[ghostsEaten_ind] += 1
                features[food_ind] = 0
                features[time_step] = 0

        if new_state.data._lose:
            features[lost_ind] = 1
            features[food_ind] = 0
            features[time_step] = 0

        #if (new_state.getPacmanPosition() in new_state.getGhostPositions()) and (new_state.data.agentStates[1].scaredTimer==0):
        #    features[lost_ind] = 1

        """
        features = [0.0]*num_feat

        features[food_ind] = state.getNumFood() - new_state.getNumFood()    # whether food was eaten in this step

        if new_state.data._win:
            features[won_ind] = 1.0
        #if (new_state.getPacmanPosition() in new_state.getGhostPositions()) and (new_state.data.agentStates[1].scaredTimer>0):
        #    features[won_ind] = 1

        for ind in range(1, len(new_state.data._eaten)):
            if new_state.data._eaten[ind]:
                features[ghostsEaten_ind] += 1

        if new_state.data._lose:
            features[lost_ind] = 1
        #if (new_state.getPacmanPosition() in new_state.getGhostPositions()) and (new_state.data.agentStates[1].scaredTimer==0):
        #    features[lost_ind] = 1

        features[time_step] = 1
        """

        return features
