####################
# TO DO: Add key pick up
# TO DO: Add way to generate more random movements from human (for IRL)
####################

import sys
from typing import Deque

import random
#from numpy.random import shuffle
sys.path.append("..")
import numpy as np
import marlgrid
import gym

from models import *

import matplotlib.pyplot as plt

#from marlgrid.rendering import InteractivePlayerWindow
from marlgrid.agents import GridAgentInterface
#from marlgrid.envs import env_from_config
from marlgrid.envs import SympathyMultiGrid

class TestRLAgent:
    def __init__(self):
        #self.player_window = InteractivePlayerWindow(
        #    caption="interactive marlgrid"
        #)
        self.episode_count = 0
        self.replay_memory = Deque()
        self.human_replay_memory = Deque()
        self.IRL_memory = Deque()

        self.params = {
            'save_file': 'save',
            'save_interval': 10,

            'training_start': 30,

            'batch_size': 16,
            'view_window_width': 5,
            'view_window_height': 5,

            'num_actions': 5,
            'num_human_actions': 5,

            'target_update': 2,
            'memory_limit': 10000,

            'no_episodes': 1000,
            'discount': 0.9,
            'eps_initial': 1, #1
            'eps_decay': 0.99 #0.95

        }

        self.pixel_scale = 65
        # Initialise value function models
        self.q_net = DQN(self.params['view_window_width'],self.params['view_window_height'], self.params['num_actions'])
        self.q_net_target = DQN(self.params['view_window_width'],self.params['view_window_height'], self.params['num_actions'])
        self.q_net_greedy = DQN(self.params['view_window_width'],self.params['view_window_height'], self.params['num_actions'])
        self.q_net_greedy_target = DQN(self.params['view_window_width'],self.params['view_window_height'], self.params['num_actions'])
        self.q_net_human =  DQN_Human(self.params['view_window_width'],self.params['view_window_height'], self.params['num_human_actions'])
        self.nextRobotstate = nextStateModel(self.params['view_window_width'],self.params['view_window_height'])
        self.nextHumanstate = nextStateModel(self.params['view_window_width'],self.params['view_window_height'])

        # Initial predicted human rewards
        self.human_rewards = np.array([0, 0, 0, 0, 0])
        self.robot_reward_vector = np.array([0,10,-1,5,-1])
        # Reward features:
        # human_pellet, robot_pellet, open_door, win_game, step

        # Initialise parameters
        self.cost_total = 0
        self.greedy_cost_total = 0
        self.cnt = 0
        self.human_cost_total = 0
        self.human_cost_before = 0
        self.cnt_human = 0
        self.IRL_error = 0
        self.robot_nextstate_error = 0
        self.human_nextstate_error = 0

        self.beta_door_value = 0
        self.beta_door_cnt = 0
        self.beta_food = 0
        self.beta_food_cnt = 0
        self.beta_win = 0
        self.beta_win_cnt = 0

    def action_step(self, obs, agent0_pos, agent1_pos, otherfood_pos, walls, door_status, agent_no):

        if agent_no == 0:

            obs = np.array(obs)/self.pixel_scale
            obs = rgb2gray(obs).reshape(1,self.params['view_window_width'],self.params['view_window_height'],1)

            # ROBOT ACTIONS
            if np.random.uniform(0,1) > self.epsilon:
                scaled_agent0_pos = agent0_pos/np.array([self.grid_width, self.grid_height])
                scaled_agent1_pos = agent1_pos/np.array([self.grid_width, self.grid_height])
                actions_robot = self.q_net.model.predict([obs, np.array([int(door_status)]).reshape(1,1), np.array([scaled_agent0_pos[0]]).reshape(1,1), np.array([scaled_agent0_pos[1]]).reshape(1,1), np.array([scaled_agent1_pos[0]]).reshape(1,1), np.array([scaled_agent1_pos[1]]).reshape(1,1)])[0]

                possible_actions = self.possibleActions(agent0_pos, walls, door_status, agent1_pos)
                actions_robot_possible = actions_robot[possible_actions]
                action_array = (possible_actions[np.argmax(actions_robot_possible)],6)

            else:

                possible_actions = self.possibleActions(agent0_pos, walls, door_status, agent1_pos)
                action_array = (np.random.choice(possible_actions),6)

        elif agent_no == 1:

            # Add epsilon to human movements
            if (np.random.uniform(0,1) < 0.2):
                possible_actions = self.possibleActions_Human(agent1_pos, walls, door_status)
                action_array = (6,np.random.choice(possible_actions))
            elif (len(otherfood_pos) == 0):
                action_array = (6,4)
            else:
                # Human action: Got forward if movement minimises distance to closest food.
                if (door_status):
                    otherfood_pos_without_room = otherfood_pos.copy()

                    possible_actions = self.possibleActions_Human(agent1_pos, walls, door_status)

                    dist_to_food = []
                    for a in possible_actions:
                        agent_tmp = agent1_pos.copy()

                        if a == 3:
                            agent_tmp[1] -= 1
                        if a == 0:
                            agent_tmp[0] += 1
                        if a == 1:
                            agent_tmp[1] += 1
                        if a == 2:
                            agent_tmp[0] -= 1


                        if door_status:
                            cur_dist = self.closestFood(agent1_pos, otherfood_pos_without_room, walls)
                            fwd_dist = self.closestFood(np.array(agent_tmp), otherfood_pos_without_room, walls)
                        else:
                            walls_door = walls.copy()
                            walls_door.append([7,2])
                            cur_dist = self.closestFood(agent1_pos, otherfood_pos_without_room, walls_door)
                            fwd_dist = self.closestFood(np.array(agent_tmp), otherfood_pos_without_room, walls_door)

                        dist_to_food.append(fwd_dist)

                    I = np.argmin(dist_to_food)
                    action_array = (6,possible_actions[I])
                else:
                    action_array = (6,4)

        return action_array

    def possibleActions_Human(self, agent_pos, walls_original, door_status):

        walls = walls_original.copy()

        if not door_status:
            walls.append([7,2])

        directions = [4] # Can always stop

        # Check if can go up
        for dir in [0,1,2,3]:
            agent_tmp = agent_pos.copy()

            if dir == 3:
                agent_tmp[1] -= 1
            if dir == 0:
                agent_tmp[0] += 1
            if dir == 1:
                agent_tmp[1] += 1
            if dir == 2:
                agent_tmp[0] -= 1

            if (agent_tmp[0] > 0) and (agent_tmp[1] > 0) and (agent_tmp[0] < (self.grid_width-1)) and (agent_tmp[1] < (self.grid_height-1)):
                if not (np.array(agent_tmp) == walls).all(1).any():
                    directions.append(dir)

        return directions

    def possibleActions(self, agent_pos, walls_original, door_status, agent1_pos):

        walls = walls_original.copy()

        if not door_status:
            walls.append([7,2])

        directions = []

        # Check if can go up
        for dir in [0,1,2,3]:
            agent_tmp = agent_pos.copy()

            if dir == 3:
                agent_tmp[1] -= 1
            if dir == 0:
                agent_tmp[0] += 1
            if dir == 1:
                agent_tmp[1] += 1
            if dir == 2:
                agent_tmp[0] -= 1

            if (agent_tmp[0] > 0) and (agent_tmp[1] > 0) and (agent_tmp[0] < (self.grid_width-1)) and (agent_tmp[1] < (self.grid_height-1)):
                if not (np.array(agent_tmp) == walls).all(1).any():
                    directions.append(dir)

        # Ability to toggle open door
        if (agent_pos == np.array([7,3])).all() and (not door_status):
            directions.append(4)

        return directions

    def closestFood(self, pos, food, walls):
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
            if any(np.array([(np.array([pos_x,pos_y]) == foods).all() for foods in food])):
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = self.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None

    def getLegalNeighbors(self, position, walls):
        # Directions
        directions = {'NORTH': (0, 1),
        'SOUTH': (0, -1),
        'EAST':  (1, 0),
        'WEST':  (-1, 0),
        'STOP':  (0, 0)}

        directionsAsList = directions.items()

        x,y = position
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        neighbors = []
        for dir, vec in directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            if next_x < 0 or next_x == self.grid_width: continue
            next_y = y_int + dy
            if next_y < 0 or next_y == self.grid_height: continue
            if not any(np.array([(np.array([next_x,next_y]) == wall).all() for wall in walls])): neighbors.append((next_x, next_y))
        return neighbors

    def save_step(self, obs_1, act, rew, obs_2, obs_3, done, reward_vector, door, door_ns, obs_2_door,won):
        #print(f"   step {self.step_count:<4d}: reward {rew} (episode total {self.cumulative_reward})")
        self.cumulative_reward += rew

        self.step_count += 1

        experience = [obs_1, act, rew, obs_3, done, reward_vector, door, door_ns, obs_2, obs_2_door,won]

        self.replay_memory.append(experience)

        # Check if too long
        if len(self.replay_memory) > self.params['memory_limit']:
            self.replay_memory.popleft()

        # Train
        if len(self.replay_memory) > self.params['training_start']:
            # Update Q-value function
            #t = time.time()
            self.train()

            if self.sympathetic_mode:
                self.robot_nextState_update()
                self.human_nextState_update()
            #print("training robot", time.time() - t)

        # Save model
        if (self.params['save_file']):
            if (self.episodesSoFar % self.params['save_interval'] == 0) and (done):
                layout_name = "GridWorld"
                name = layout_name + '_Empathic_beta_' + str(int(not self.sympathetic_mode))
                save_location = 'saves/' + name + "_" + str(self.episodesSoFar)
                self.q_net.model.save(save_location)

    def save_human_step(self, obs, act, nextobs, done, reward_vector, door, door_ns):

        if self.sympathetic_mode:
            experience = [obs, act, nextobs, done, reward_vector, door, door_ns]
            self.human_replay_memory.append(experience)

            if len(self.IRL_memory) == 0:
                self.IRL_memory.append(experience)
            else:
                inside = False
                for a in range(len(self.IRL_memory)):
                    if ((experience[0][0][1] == self.IRL_memory[a][0][0][1]).all()) and (experience[0][2] == self.IRL_memory[a][0][2]).all() and (experience[5] == self.IRL_memory[a][5]) and (experience[1] == self.IRL_memory[a][1]):
                        inside = True
                    if inside == True:
                        break

                if inside == False:
                    self.IRL_memory.append(experience)

            # Check if too long
            if len(self.human_replay_memory) > self.params['memory_limit']:
                self.human_replay_memory.popleft()

            # Check if too long
            if len(self.IRL_memory) > self.params['memory_limit']:
                self.IRL_memory.popleft()

            if len(self.human_replay_memory) > self.params['training_start']:
                self.train_human()

            if len(self.IRL_memory) > self.params['training_start']:
                self.human_reward_prediction()


    def robot_nextState_update(self):

        batch = random.sample(self.replay_memory, self.params['batch_size'])

        # Full information of state
        obs_full = extract(batch,0)

        # Extract just the field of vision for robot
        obs = extract(obs_full,0)
        obs_r = extract(obs,0)
        obs_r = [np.array(i)/self.pixel_scale for i in obs_r]
        obs_r = rgb2gray(obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        obs_h = extract(obs,1)
        obs_h = [np.array(i)/self.pixel_scale for i in obs_h]
        obs_h = rgb2gray(obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_pos = extract(obs_full,1)
        r_pos_x = np.array(extract(r_pos,0)).reshape(-1,1)
        r_pos_y = np.array(extract(r_pos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_pos = extract(obs_full,2)
        h_pos_x = np.array(extract(h_pos,0)).reshape(-1,1)
        h_pos_y = np.array(extract(h_pos,1)).reshape(-1,1)

        actions = np.array(extract(batch,1))

        # Full information of state
        next_obs_fill = extract(batch,8)

        # Extract just the field of vision for robot
        next_obs = extract(next_obs_fill,0)
        next_obs_r = extract(next_obs,0)
        next_obs_r = [np.array(i)/self.pixel_scale for i in next_obs_r]
        next_obs_r = rgb2gray(next_obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        next_obs_h = extract(next_obs,1)
        next_obs_h = [np.array(i)/self.pixel_scale for i in next_obs_h]
        next_obs_h = rgb2gray(next_obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_nextpos = extract(next_obs_fill,1)
        r_nextpos_x = np.array(extract(r_nextpos,0)).reshape(-1,1)
        r_nextpos_y = np.array(extract(r_nextpos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_nextpos = extract(next_obs_fill,2)
        h_nextpos_x = np.array(extract(h_nextpos,0)).reshape(-1,1)
        h_nextpos_y = np.array(extract(h_nextpos,1)).reshape(-1,1)

        terminal = extract(batch,10)

        door_s = np.array(extract(batch, 6)).reshape(-1,1)
        door_ns = np.array(extract(batch, 9)).reshape(-1,1)

        batch_target = np.concatenate((door_ns.reshape(self.params['batch_size'],1), r_nextpos_x.reshape(self.params['batch_size'],1), r_nextpos_y.reshape(self.params['batch_size'],1), h_nextpos_x.reshape(self.params['batch_size'],1), h_nextpos_y.reshape(self.params['batch_size'],1)),axis=1)
        batch_target = np.concatenate((next_obs_r.reshape(self.params['batch_size'],-1),batch_target),axis=1)

        self.nextRobotstate.model.fit([obs_r, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y, actions], batch_target, shuffle=True, epochs=10, batch_size=self.params['batch_size'],verbose=0)

        error = self.nextRobotstate.model.history.history['loss'][-1]

        self.robot_nextstate_error += error

    def human_nextState_update(self):

        batch = random.sample(self.replay_memory, self.params['batch_size'])

        # Full information of state
        obs_full = extract(batch,0)

        # Extract just the field of vision for robot
        obs = extract(obs_full,0)
        obs_r = extract(obs,0)
        obs_r = [np.array(i)/self.pixel_scale for i in obs_r]
        obs_r = rgb2gray(obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        obs_h = extract(obs,1)
        obs_h = [np.array(i)/self.pixel_scale for i in obs_h]
        obs_h = rgb2gray(obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_pos = extract(obs_full,1)
        r_pos_x = np.array(extract(r_pos,0)).reshape(-1,1)
        r_pos_y = np.array(extract(r_pos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_pos = extract(obs_full,2)
        h_pos_x = np.array(extract(h_pos,0)).reshape(-1,1)
        h_pos_y = np.array(extract(h_pos,1)).reshape(-1,1)

        actions = np.array(extract(batch,1))

        # Full information of state
        next_obs_fill = extract(batch,8)

        # Extract just the field of vision for robot
        next_obs = extract(next_obs_fill,0)
        next_obs_r = extract(next_obs,0)
        next_obs_r = [np.array(i)/self.pixel_scale for i in next_obs_r]
        next_obs_r = rgb2gray(next_obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        next_obs_h = extract(next_obs,1)
        next_obs_h = [np.array(i)/self.pixel_scale for i in next_obs_h]
        next_obs_h = rgb2gray(next_obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_nextpos = extract(next_obs_fill,1)
        r_nextpos_x = np.array(extract(r_nextpos,0)).reshape(-1,1)
        r_nextpos_y = np.array(extract(r_nextpos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_nextpos = extract(next_obs_fill,2)
        h_nextpos_x = np.array(extract(h_nextpos,0)).reshape(-1,1)
        h_nextpos_y = np.array(extract(h_nextpos,1)).reshape(-1,1)

        terminal = extract(batch,10)

        door_s = np.array(extract(batch, 6)).reshape(-1,1)
        door_ns = np.array(extract(batch, 9)).reshape(-1,1)

        batch_target = np.concatenate((door_ns.reshape(self.params['batch_size'],1), r_nextpos_x.reshape(self.params['batch_size'],1), r_nextpos_y.reshape(self.params['batch_size'],1), h_nextpos_x.reshape(self.params['batch_size'],1), h_nextpos_y.reshape(self.params['batch_size'],1)),axis=1)
        batch_target = np.concatenate((next_obs_h.reshape(self.params['batch_size'],-1),batch_target),axis=1)

        self.nextHumanstate.model.fit([obs_h, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y, actions], batch_target, shuffle=True, epochs=10, batch_size=self.params['batch_size'],verbose=0)

        error = self.nextHumanstate.model.history.history['loss'][-1]

        self.human_nextstate_error += error

    def train(self):

        def QvalueFunction_setting(numerator_Qg, numerator_Qp, Q_differences):

            c = 1
            x = c*(numerator_Qg-numerator_Qp)/max(Q_differences)

            B1 = 1/(1+np.exp(-1*x))
            B2 = (1-B1)

            return B1, B2

        if sum(self.human_rewards) == 0:
            l1_scale = 1
        else:
            l1_scale = np.sum(self.robot_reward_vector)/np.sum(np.abs(self.human_rewards))

        batch = random.sample(self.replay_memory, self.params['batch_size'])

        # Full information of state
        obs_full = extract(batch,0)

        # Extract just the field of vision for robot
        obs = extract(obs_full,0)
        obs_r = extract(obs,0)
        obs_r = [np.array(i)/self.pixel_scale for i in obs_r]
        obs_r = rgb2gray(obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        obs_h = extract(obs,1)
        obs_h = [np.array(i)/self.pixel_scale for i in obs_h]
        obs_h = rgb2gray(obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_pos = extract(obs_full,1)
        r_pos_x = np.array(extract(r_pos,0)).reshape(-1,1)
        r_pos_y = np.array(extract(r_pos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_pos = extract(obs_full,2)
        h_pos_x = np.array(extract(h_pos,0)).reshape(-1,1)
        h_pos_y = np.array(extract(h_pos,1)).reshape(-1,1)

        actions = extract(batch,1)
        rewards = np.array(extract(batch,2))

        reward_human_vector = np.array(extract(batch,5))
        reward_human = np.array(extract(batch,5))
        reward_human = l1_scale*np.dot(reward_human,self.human_rewards)

        # Full information of state (s3)
        next_obs_fill = extract(batch,3)

        # Extract just the field of vision for robot
        next_obs = extract(next_obs_fill,0)
        next_obs_r = extract(next_obs,0)
        next_obs_r = [np.array(i)/self.pixel_scale for i in next_obs_r]
        next_obs_r = rgb2gray(next_obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        next_obs_h = extract(next_obs,1)
        next_obs_h = [np.array(i)/self.pixel_scale for i in next_obs_h]
        next_obs_h = rgb2gray(next_obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_nextpos = extract(next_obs_fill,1)
        r_nextpos_x = np.array(extract(r_nextpos,0)).reshape(-1,1)
        r_nextpos_y = np.array(extract(r_nextpos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_nextpos = extract(next_obs_fill,2)
        h_nextpos_x = np.array(extract(h_nextpos,0)).reshape(-1,1)
        h_nextpos_y = np.array(extract(h_nextpos,1)).reshape(-1,1)

        # Full information of state (s2)
        next_obs_fill2 = extract(batch,8)

        # Extract just the field of vision for robot
        next_obs2 = extract(next_obs_fill2,0)
        next_obs_r2 = extract(next_obs2,0)
        next_obs_r2 = [np.array(i)/self.pixel_scale for i in next_obs_r2]
        next_obs_r2 = rgb2gray(next_obs_r2).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        next_obs_h2 = extract(next_obs2,1)
        next_obs_h2 = [np.array(i)/self.pixel_scale for i in next_obs_h2]
        next_obs_h2 = rgb2gray(next_obs_h2).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_nextpos2 = extract(next_obs_fill2,1)
        r_nextpos_x2 = np.array(extract(r_nextpos2,0)).reshape(-1,1)
        r_nextpos_y2 = np.array(extract(r_nextpos2,1)).reshape(-1,1)

        # Extract x and y position of human
        h_nextpos2 = extract(next_obs_fill2,2)
        h_nextpos_x2 = np.array(extract(h_nextpos2,0)).reshape(-1,1)
        h_nextpos_y2 = np.array(extract(h_nextpos2,1)).reshape(-1,1)

        terminal = extract(batch,10)

        door_s = np.array(extract(batch, 6)).reshape(-1,1)
        door_ns2 = np.array(extract(batch, 9)).reshape(-1,1)
        door_ns = np.array(extract(batch, 7)).reshape(-1,1)

        # Next possible actions
        next_actions = [self.possibleActions(r_nextpos[i]*[self.grid_width,self.grid_height], self.walls, door_ns[i], h_nextpos[i]*[self.grid_width,self.grid_height]) for i in range(self.params['batch_size'])]

        # Next state q-value
        q_vals = self.q_net.model.predict([obs_r, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y])
        q_vals_next = self.q_net_target.model.predict([next_obs_r, door_ns, r_nextpos_x, r_nextpos_y, h_nextpos_x, h_nextpos_y])
        q_vals_next = [np.max(q_vals_next[i][next_actions[i]]) for i in range(self.params['batch_size'])]#np.max(q_vals_next[next_actions],axis=1)

        # Greedy agent next values
        if self.sympathetic_mode:
            q_vals_greedy = self.q_net_greedy.model.predict([obs_r, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y])
            q_vals_greedy_next = self.q_net_greedy_target.model.predict([next_obs_r, door_ns, r_nextpos_x, r_nextpos_y, h_nextpos_x, h_nextpos_y])
            q_vals_greedy_next = [np.max(q_vals_greedy_next[i][next_actions[i]]) for i in range(self.params['batch_size'])]

        B1_val = []
        B2_val = []

        for i in range(self.params['batch_size']):

            denominator_differences = []

            # Beta value
            if (not self.sympathetic_mode):
                B1 = 1
                B2 = 0
            else:
                if terminal[i]:
                    numerator_Qg = l1_scale*float(max(self.q_net_human.model.predict([next_obs_h2[i].reshape(1,self.params['view_window_width'],self.params['view_window_height'],1), np.array(door_ns2[i]).reshape(1,1), np.array(r_nextpos_x2[i]).reshape(1,1), np.array(r_nextpos_y2[i]).reshape(1,1), np.array(h_nextpos_x2[i]).reshape(1,1), np.array(h_nextpos_y2[i]).reshape(1,1)])[0]))
                    numerator_Qp = 0

                else:
                    numerator_Qg = l1_scale*float(max(self.q_net_human.model.predict([next_obs_h2[i].reshape(1,self.params['view_window_width'],self.params['view_window_height'],1), np.array(door_ns2[i]).reshape(1,1), np.array(r_nextpos_x2[i]).reshape(1,1), np.array(r_nextpos_y2[i]).reshape(1,1), np.array(h_nextpos_x2[i]).reshape(1,1), np.array(h_nextpos_y2[i]).reshape(1,1)])[0]))
                    numerator_Qp = float(max(self.q_net_greedy.model.predict([next_obs_r2[i].reshape(1,self.params['view_window_width'],self.params['view_window_height'],1), np.array(door_ns2[i]).reshape(1,1), np.array(r_nextpos_x2[i]).reshape(1,1), np.array(r_nextpos_y2[i]).reshape(1,1), np.array(h_nextpos_x2[i]).reshape(1,1), np.array(h_nextpos_y2[i]).reshape(1,1)])[0]))

                possible_actions = self.possibleActions(r_pos[i]*[self.grid_width,self.grid_height], self.walls, door_s[i], h_pos[i]*[self.grid_width,self.grid_height])

                input_RobotNext = [obs_r[i].reshape(self.params['view_window_width'],self.params['view_window_height'],1), np.array(door_s[i]).reshape(1,), np.array(r_pos_x[i]).reshape(1,), np.array(r_pos_y[i]).reshape(1,), np.array(h_pos_x[i]).reshape(1,), np.array(h_pos_y[i]).reshape(1,)]
                input_HumanNext = [obs_h[i].reshape(self.params['view_window_width'],self.params['view_window_height'],1), np.array(door_s[i]).reshape(1,), np.array(r_pos_x[i]).reshape(1,), np.array(r_pos_y[i]).reshape(1,), np.array(h_pos_x[i]).reshape(1,), np.array(h_pos_y[i]).reshape(1,)]

                input_RobotNext = [np.array([k,]*len(possible_actions)) for k in input_RobotNext]
                input_HumanNext = [np.array([k,]*len(possible_actions)) for k in input_HumanNext]

                input_RobotNext.append(np.array(possible_actions))
                input_HumanNext.append(np.array(possible_actions))

                s_next_pred_model = np.array(self.nextRobotstate.model.predict(input_RobotNext))
                s_next_ghost_pred_model = np.array(self.nextHumanstate.model.predict(input_HumanNext))

                s_next_window = s_next_pred_model[:,:-5].reshape(-1,self.params['view_window_width'],self.params['view_window_height'])
                for action_int in range(len(possible_actions)):
                    a_int = possible_actions[action_int]
                    if a_int == 2: #west
                        s_next_window[action_int,:,0] = np.random.uniform(0.0, 1.0, s_next_window[action_int,:,0].shape)
                    elif a_int == 3: #North
                        s_next_window[action_int,0,:] = np.random.uniform(0.0, 1.0, s_next_window[action_int,0,:].shape)
                    elif a_int == 1: #South
                        s_next_window[action_int,-1,:] = np.random.uniform(0.0, 1.0, s_next_window[action_int,-1,:].shape)
                    elif a_int == 0: #East
                        s_next_window[action_int,:,-1] = np.random.uniform(0.0, 1.0, s_next_window[action_int,:,-1].shape)

                rs = s_next_window.reshape(-1,self.params['view_window_width'],self.params['view_window_height'],1)
                r_d = s_next_pred_model[:,-5].reshape(-1,1)
                r_rposx = s_next_pred_model[:,-4].reshape(-1,1)
                r_rposy = s_next_pred_model[:,-3].reshape(-1,1)
                r_hposx = s_next_pred_model[:,-2].reshape(-1,1)
                r_hposy = s_next_pred_model[:,-1].reshape(-1,1)

                hs = s_next_ghost_pred_model[:,:-5].reshape(-1,self.params['view_window_width'],self.params['view_window_height'],1)
                h_d = s_next_ghost_pred_model[:,-5].reshape(-1,1)
                h_rposx = s_next_ghost_pred_model[:,-4].reshape(-1,1)
                h_rposy = s_next_ghost_pred_model[:,-3].reshape(-1,1)
                h_hposx = s_next_ghost_pred_model[:,-2].reshape(-1,1)
                h_hposy = s_next_ghost_pred_model[:,-1].reshape(-1,1)

                denominator_Qg = l1_scale*np.max(self.q_net_human.model.predict([hs,h_d,h_rposx,h_rposy,h_hposx,h_hposy]),axis=1)
                denominator_Qp = np.max(self.q_net_greedy.model.predict([rs,r_d,r_rposx,r_rposy,r_hposx,r_hposy]),axis=1)

                denominator_differences = np.abs(denominator_Qg-denominator_Qp)

                B1, B2 = QvalueFunction_setting(numerator_Qg, numerator_Qp, denominator_differences)

                if reward_human_vector[i][2] == 1:
                    self.beta_door_value += B1
                    self.beta_door_cnt += 1

                if reward_human_vector[i][1] == 1:
                    self.beta_food += B1
                    self.beta_food_cnt += 1

                if reward_human_vector[i][3] == 1:
                    self.beta_win += B1
                    self.beta_win_cnt += 1

            B1_val.append(B1)
            B2_val.append(B2)

            R_symp = B1*rewards[i] + B2*reward_human[i]

            if terminal[i] == False:
                q_vals[i][actions[i]] = R_symp + self.params['discount']*q_vals_next[i]
                if self.sympathetic_mode:
                    q_vals_greedy[i][actions[i]] = rewards[i] + self.params['discount']*q_vals_greedy_next[i]
            else:
                q_vals[i][actions[i]] = R_symp
                if self.sympathetic_mode:
                    q_vals_greedy[i][actions[i]] = rewards[i]

        self.q_net.model.fit([obs_r, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y], q_vals, shuffle=True, epochs = 5, batch_size = self.params['batch_size'], verbose = 0)
        if self.sympathetic_mode:
            self.q_net_greedy.model.fit([obs_r, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y], q_vals_greedy, shuffle=True, epochs = 5, batch_size = self.params['batch_size'], verbose = 0)

        error = self.q_net.model.history.history['loss'][-1]
        if self.sympathetic_mode:
            error_greedy = self.q_net_greedy.model.history.history['loss'][-1]
            self.greedy_cost_total += error_greedy

        self.cost_total += error
        self.cnt += 1

    def train_human(self):

        def TD_error(obs,actions, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y):

            # Calculate TD error:
            q_vals = np.array(self.q_net_human.model.predict([obs, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y]))
            perc_correct = np.sum(np.argmax(q_vals,1) == actions)/len(actions)

            return perc_correct

        batch = random.sample(self.human_replay_memory, self.params['batch_size'])

        obs_full = extract(batch,0)

        obs = extract(obs_full,0)
        obs_r = extract(obs,0)
        obs_r = [np.array(i)/self.pixel_scale for i in obs_r]
        obs_r = rgb2gray(obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        obs_h = extract(obs,1)
        obs_h = [np.array(i)/self.pixel_scale for i in obs_h]
        obs_h = rgb2gray(obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_pos = extract(obs_full,1)
        r_pos_x = np.array(extract(r_pos,0)).reshape(-1,1)
        r_pos_y = np.array(extract(r_pos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_pos = extract(obs_full,2)
        h_pos_x = np.array(extract(h_pos,0)).reshape(-1,1)
        h_pos_y = np.array(extract(h_pos,1)).reshape(-1,1)

        actions = extract(batch,1)

        # Full information of state
        next_obs_fill = extract(batch,2)

        # Extract just the field of vision for robot
        next_obs = extract(next_obs_fill,0)
        next_obs_r = extract(next_obs,0)
        next_obs_r = [np.array(i)/self.pixel_scale for i in next_obs_r]
        next_obs_r = rgb2gray(next_obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        next_obs_h = extract(next_obs,1)
        next_obs_h = [np.array(i)/self.pixel_scale for i in next_obs_h]
        next_obs_h = rgb2gray(next_obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_nextpos = extract(next_obs_fill,1)
        r_nextpos_x = np.array(extract(r_nextpos,0)).reshape(-1,1)
        r_nextpos_y = np.array(extract(r_nextpos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_nextpos = extract(next_obs_fill,2)
        h_nextpos_x = np.array(extract(h_nextpos,0)).reshape(-1,1)
        h_nextpos_y = np.array(extract(h_nextpos,1)).reshape(-1,1)

        terminal = extract(batch,3)

        door_s = np.array(extract(batch,5)).reshape(-1,1)
        door_ns = np.array(extract(batch,6)).reshape(-1,1)

        # Convert actions to onehot encoding
        q_vals = np.zeros((self.params['batch_size'],self.params['num_human_actions']))

        pred_q_vals = np.array(self.q_net_human.model.predict([obs_h, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y]))

        for i in range(self.params['batch_size']):
            for r in range(self.params['num_human_actions']):
                if (np.argmax(pred_q_vals[i]) == actions[i]) and (r==actions[i]):
                    q_vals[i][r] = 1 #pred_q_vals[i][r]
                elif (r == actions[i]):
                    q_vals[i][actions[i]] = 1
                else:
                    q_vals[i][r] = 0 #pred_q_vals[i][r]

        self.q_net_human.model.fit([obs_h, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y], q_vals, shuffle=True, epochs=10, batch_size=self.params['batch_size'],verbose=0)

        perc_error = TD_error(obs_h, actions, door_s, r_pos_x, r_pos_y, h_pos_x, h_pos_y)

        self.human_cost_total += perc_error
        self.cnt_human += 1

    def human_reward_prediction(self):

        batch = random.sample(self.IRL_memory, self.params['batch_size'])

        obs_full = extract(batch,0)

        obs = extract(obs_full,0)
        obs_r = extract(obs,0)
        obs_r = [np.array(i)/self.pixel_scale for i in obs_r]
        obs_r = rgb2gray(obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        obs_h = extract(obs,1)
        obs_h = [np.array(i)/self.pixel_scale for i in obs_h]
        obs_h = rgb2gray(obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_pos = extract(obs_full,1)
        r_pos_x = np.array(extract(r_pos,0)).reshape(-1,1)
        r_pos_y = np.array(extract(r_pos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_pos = extract(obs_full,2)
        h_pos_x = np.array(extract(h_pos,0)).reshape(-1,1)
        h_pos_y = np.array(extract(h_pos,1)).reshape(-1,1)

        actions = extract(batch,1)

        # Full information of state
        next_obs_fill = extract(batch,2)

        # Extract just the field of vision for robot
        next_obs = extract(next_obs_fill,0)
        next_obs_r = extract(next_obs,0)
        next_obs_r = [np.array(i)/self.pixel_scale for i in next_obs_r]
        next_obs_r = rgb2gray(next_obs_r).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)
        next_obs_h = extract(next_obs,1)
        next_obs_h = [np.array(i)/self.pixel_scale for i in next_obs_h]
        next_obs_h = rgb2gray(next_obs_h).reshape(self.params['batch_size'],self.params['view_window_width'],self.params['view_window_height'],1)

        # Extract x and y position of robot
        r_nextpos = extract(next_obs_fill,1)
        r_nextpos_x = np.array(extract(r_nextpos,0)).reshape(-1,1)
        r_nextpos_y = np.array(extract(r_nextpos,1)).reshape(-1,1)

        # Extract x and y position of human
        h_nextpos = extract(next_obs_fill,2)
        h_nextpos_x = np.array(extract(h_nextpos,0)).reshape(-1,1)
        h_nextpos_y = np.array(extract(h_nextpos,1)).reshape(-1,1)

        terminal = extract(batch,3)

        reward_vector = extract(batch,4)

        door_s = np.array(extract(batch,5)).reshape(-1,1)
        door_ns = np.array(extract(batch,6)).reshape(-1,1)

        R_target = []
        rewardfeaturelist = []

        for i in range(self.params['batch_size']):

            q = self.q_net_human.model_bsm.predict([obs_h[i].reshape(1,self.params['view_window_width'],self.params['view_window_height'],1), door_s[i].reshape(1,1), r_pos_x[i].reshape(1,1), r_pos_y[i].reshape(1,1), h_pos_x[i].reshape(1,1), h_pos_y[i].reshape(1,1)])
            q = float(q[0][actions[i]])

            if terminal[i] == True:
                q_max = 0.0
            else:
                q_max = float(max(self.q_net_human.model_bsm.predict([next_obs_h[i].reshape(1,self.params['view_window_width'],self.params['view_window_height'],1), door_ns[i].reshape(1,1), r_nextpos_x[i].reshape(1,1), r_nextpos_y[i].reshape(1,1), h_nextpos_x[i].reshape(1,1), h_nextpos_y[i].reshape(1,1)])[0]))

            R = q - self.params['discount']*q_max

            R_target.append(R)
            rewardfeaturelist.append(reward_vector[i])

            pred_R = np.dot(self.human_rewards, reward_vector[i])

            difference_R = (pred_R - R)
            self.human_rewards = self.human_rewards - (0.2 * reward_vector[i] * difference_R)

        # Rough estimate of reward error
        error = 0
        for i in range(self.params['batch_size']):
            error +=  np.sum(np.abs(np.dot(self.human_rewards, rewardfeaturelist[i]) - R_target[i]))

        self.IRL_error += error

    def reward_features(self, foodbefore_0, foodbefore_1, foodafter_0, foodafter_1, door_status, last_door):
        # human_pellet, robot_pellet, opening door, finishing game, step

        reward_vector = np.array([0, 0, 0, 0, 0])

        # Check if human has eaten pellet
        if len(foodafter_1) < len(foodbefore_1):
            reward_vector[0] = 1

        # Check if robot has eaten pellet
        if len(foodafter_0) < len(foodbefore_0):
            reward_vector[1] = 1

        # Check if door has been opened for the first time
        if door_status:
            if last_door == False:
                reward_vector[2] = 1

        # Check if game is finished
        if len(foodafter_0) == 0:
            reward_vector[3] = 1

        # If none of the above occurs, then just taken a step
        if sum(reward_vector) == 0:
            reward_vector[4] = 1

        return reward_vector

    def start_episode(self):
        self.cumulative_reward = 0
        self.step_count = 0

    def end_episode(self):
        print(
            f"Finished episode {self.episode_count} after {self.step_count} steps."
            f"  Episode return was {self.cumulative_reward}."
            f"  Epsilon was {self.epsilon}."
        )
        self.episode_count += 1

# Other useful functions
def extract(lst, pos):
    return [item[pos] for item in lst]

def rgb2gray(rgb):
    rgb_converted = np.dot(rgb, [0.2989, 0.5870, 0.1140])
    rgb_converted = 2*rgb_converted
    rgb_converted[rgb_converted > 1] = 1
    return rgb_converted

# Create the environment based on the combined env/player config
env = gym.make('MarlGrid-AgentSympathy15x15-v0')

# Create a human player interface per the class defined above
agents = TestRLAgent()
agents.sympathetic_mode = True

agents.grid_width = env.grid.width
agents.grid_height = env.grid.height

for episodes in range(agents.params['no_episodes']):

    print("Episode:", str(episodes))

    # Start an episode!
    # Each observation from the environment contains a list of observaitons for each agent.
    # In this case there's only one agent so the list will be of length one.
    obs_list = env.reset()

    agents.walls = env.walls.copy()

    agents.start_episode()
    done = False
    won = False
    human_done = False
    stop_human_training = False

    agents.episodesSoFar = episodes

    lastR_state = None
    lastR_door = None
    lastH_state = None
    lastH_door = None

    robot_action = None
    human_action = None
    lastR_food = None
    lastH_food = None

    door_was_opened = False

    food_obs1 = [None, None]
    food_obs2 = [None, None]
    food_obs3 = [None, None]

    while not done:

        for agent_no in [0,1]:

            #if door_was_opened:
            env.render() # OPTIONAL: render the whole scene + birds eye view

            agents.epsilon = max(0.1,agents.params['eps_initial']*(agents.params['eps_decay']**episodes))

            # Get location of all available food (human)
            humanfood_pos = env.agents[1].foodpos

            # Check whether door is open
            door_status = env.agents[0].door_opened

            if door_status:
                door_was_opened = True

            if (agent_no == 0) and (lastH_state is not None):
                # Save state, action, reward and next state for robot
                reward_vector = agents.reward_features(lastR_food[0], lastR_food[1], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[0].door_opened, lastR_door)
                if env.agents[0].door_opened:
                    if lastR_door:
                        reward_vector[2] = 0
                reward_vector[4] = 1

                # reward calculation
                # human_food, robot_food, opening door, finishing game, step
                robot_reward = np.dot(reward_vector,agents.robot_reward_vector)

                reward_vector = agents.reward_features(lastR_food[0], lastR_food[1], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[0].door_opened, lastR_door)

                current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height])]

                agents.save_step(
                    lastR_state, robot_action, robot_reward, lastH_state, current_state, done, reward_vector, int(lastR_door), int(env.agents[0].door_opened), lastH_door, won
                )

            if not stop_human_training:
                if (agent_no == 1) and (lastH_state is not None):
                    # Save state, action, reward and next state for robot
                    reward_vector = agents.reward_features(lastH_food[0], lastH_food[1], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[0].door_opened, lastH_door)

                    current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height])]

                    agents.save_human_step(
                        lastH_state, human_action, current_state, human_done, reward_vector, int(lastH_door), int(env.agents[0].door_opened)
                    )

                    if human_done:
                        stop_human_training = True

            # Output action for each agent and whether robot has picked up key
            action_array = agents.action_step(obs_list[0], env.agents[0].pos, env.agents[1].pos, humanfood_pos, agents.walls, door_status,agent_no)

            if agent_no == 0:

                lastR_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height])]
                lastR_door = door_status
                robot_action = action_array[0]
                lastR_food = [env.agents[0].foodpos, env.agents[1].foodpos]

            elif agent_no == 1:

                lastH_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height])]
                lastH_door = door_status
                human_action = action_array[1]
                lastH_food = [env.agents[0].foodpos, env.agents[1].foodpos]

            # Simulate next action (code adapted to moving in the direction indicated.)
            # action = 0 (east), action = 1 (south), action = 2 (west), action = 3 (north)
            if action_array[agent_no] == 4:
                if agent_no == 0:
                    if env.agents[agent_no].dir == 3:
                        action_array_tmp = (3,action_array[1])
                    else:
                        while env.agents[agent_no].dir != 3:
                            action_array_tmp = (0,action_array[1])
                            next_obs_list, rew_list, done, _ = env.step(action_array_tmp)

                        action_array_tmp = (3,action_array[1])
                        next_obs_list, rew_list, done, _ = env.step(action_array_tmp)

                else:
                    action_array_tmp = (action_array[0],6)

                next_obs_list, rew_list, done, _ = env.step(action_array_tmp)

            elif action_array[agent_no] == env.agents[agent_no].dir:
                if agent_no == 0:
                    action_array_tmp = (2,action_array[1])
                else:
                    action_array_tmp = (action_array[0],2)
                next_obs_list, rew_list, done, _ = env.step(action_array_tmp)
            else:
                while action_array[agent_no] != env.agents[agent_no].dir:
                    if agent_no == 0:
                        action_array_tmp = (0,action_array[1])
                    else:
                        action_array_tmp = (action_array[0],0)
                    next_obs_list, rew_list, done, _ = env.step(action_array_tmp)

                if action_array[agent_no] == env.agents[agent_no].dir:
                    if agent_no == 0:
                        action_array_tmp = (2,action_array[1])
                    else:
                        action_array_tmp = (action_array[0],2)
                    next_obs_list, rew_list, done, _ = env.step(action_array_tmp)

            # If robot has finished all of its own food, game will end
            robotfood_pos = env.agents[0].foodpos
            if len(robotfood_pos) == 0:
                done = True
                won = True

            human_food_remaining = env.agents[1].foodpos
            if len(human_food_remaining) == 0:
                human_done = True

            obs_list = next_obs_list.copy()

            # Check if game ended
            if done:
                if (agent_no == 0):
                    # Save state, action, reward and next state for robot
                    reward_vector = agents.reward_features(lastR_food[0], lastR_food[1], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[0].door_opened, lastR_door)
                    if env.agents[0].door_opened:
                        if lastR_door:
                            reward_vector[2] = 0
                    reward_vector[4] = 1

                    # reward calculation
                    # human_food, robot_food, opening door, finishing game, step
                    robot_reward = np.dot(reward_vector,agents.robot_reward_vector)

                    reward_vector = agents.reward_features(lastR_food[0], lastR_food[1], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[0].door_opened, lastR_door)

                    current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height])]

                    agents.save_step(
                        lastR_state, robot_action, robot_reward, current_state, current_state, done, reward_vector, int(lastR_door), int(env.agents[0].door_opened), int(env.agents[0].door_opened), won
                    )

                    # Save state, action, reward and next state for robot
                    reward_vector = agents.reward_features(lastH_food[0], lastH_food[1], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[0].door_opened, lastH_door)

                    current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height])]
                    if not stop_human_training:
                        agents.save_human_step(
                            lastH_state, human_action, current_state, human_done, reward_vector, int(lastH_door), int(env.agents[0].door_opened)
                        )

                elif (agent_no == 1):

                    # Save state, action, reward and next state for robot
                    reward_vector = agents.reward_features(lastR_food[0], lastR_food[1], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[0].door_opened, lastR_door)
                    if env.agents[0].door_opened:
                        if lastR_door:
                            reward_vector[2] = 0
                    reward_vector[4] = 1

                    # reward calculation
                    # human_food, robot_food, opening door, finishing game, step
                    robot_reward = np.dot(reward_vector,agents.robot_reward_vector)

                    reward_vector = agents.reward_features(lastR_food[0], lastR_food[1], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[0].door_opened, lastR_door)

                    current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height])]

                    agents.save_step(
                        lastR_state, robot_action, robot_reward, lastH_state, current_state, done, reward_vector, int(lastR_door), int(env.agents[0].door_opened), lastH_door, won
                    )

                    # Save state, action, reward and next state for robot
                    reward_vector = agents.reward_features(lastH_food[0], lastH_food[1], env.agents[0].foodpos, env.agents[1].foodpos, env.agents[0].door_opened, lastH_door)

                    current_state = [obs_list, env.agents[0].pos/np.array([agents.grid_width, agents.grid_height]), env.agents[1].pos/np.array([agents.grid_width, agents.grid_height])]

                    if not stop_human_training:
                        agents.save_human_step(
                            lastH_state, human_action, current_state, human_done, reward_vector, int(lastH_door), int(env.agents[0].door_opened)
                        )

                break

    agents.end_episode()

    # Update Q-value target
    if episodes % agents.params['target_update'] == 0:
        agents.q_net_target.model.set_weights(agents.q_net.model.get_weights())
        agents.q_net_greedy_target.model.set_weights(agents.q_net_greedy.model.get_weights())

    # Print out episode metrics
    if agents.cnt_human == 0:
        agents.cnt_human = 1
    if agents.cnt == 0:
        agents.cnt = 1
    if agents.beta_door_cnt == 0:
        agents.beta_door_cnt = 1
    if agents.beta_food_cnt == 0:
        agents.beta_food_cnt = 1
    if agents.beta_win_cnt == 0:
        agents.beta_win_cnt = 1

    log_file = open('./logs/'+'GreedyAgent' + str(int(not agents.sympathetic_mode)) +'.log','a')
    log_file.write("# %4d | steps_t: %5d |r: %5d | cost: %6f | cost_greedy: %6f | cost_human: %6f | e: %10f | robot_foodleft: %2d | H_foodleft: %2d | door_opened: %r | won: %r \n" %
                            (episodes, agents.cnt, agents.cumulative_reward, float(agents.cost_total)/float(agents.cnt), float(agents.greedy_cost_total)/float(agents.cnt), float(agents.human_cost_total)/float(agents.cnt_human), agents.epsilon, len(env.agents[0].foodpos), len(env.agents[1].foodpos), door_was_opened, won))

    # Print out episode metrics
    if agents.sympathetic_mode:
        log_file = open('./logs/'+'IRL_rewards'+'.log','a')
        log_file.write("# %4d | reward: %r \n" %
                                    (episodes,list(agents.human_rewards)))

    # Print out beta at door opening
    if agents.sympathetic_mode:
        log_file = open('./logs/'+'beta_door'+'.log','a')
        log_file.write("# %4d | beta: %r | beta_food: %r | beta_win: %r \n" %
                                    (episodes,float(agents.beta_door_value)/float(agents.beta_door_cnt),float(agents.beta_food)/float(agents.beta_food_cnt),float(agents.beta_win)/float(agents.beta_win_cnt)))

    agents.cnt = 0
    agents.greedy_cost_total = 0
    agents.cost_total = 0
    agents.cnt_human = 0
    agents.human_cost_total = 0
    agents.human_cost_before = 0

    agents.beta_door_value = 0
    agents.beta_door_cnt = 0
    agents.beta_food = 0
    agents.beta_food_cnt = 0
    agents.beta_win = 0
    agents.beta_win_cnt = 0
    agents.sa_reward_error = 0


