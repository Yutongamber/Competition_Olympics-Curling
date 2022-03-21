# -*- coding:utf-8  -*-
# Time  : 2022/3/27 下午: 16:00
# Author: Yutongamber


import numpy as np
import math
import random

IDX_TO_COLOR = {
    0: 'light green',
    1: 'green',
    2: 'sky blue',
    3: 'yellow',
    4: 'grey',
    5: 'purple',
    6: 'black',
    7: 'red',
    8: 'blue'
}


team_id = {
    0: 'purple',
    1: 'green'
}

COLOR_TO_IDX = {
    'purple': 5,
    'green': 1
}


class rule_agent:
    def __init__(self):
        self.count = 0
        self.obs = None
        self.ball_left = None
        self.ball_first = 0
        self.mark = 30 # hyper
        self.contrl_player_idx = None
        self.ball_oppo_pos = []
        self.action_mark = None
        self.actions_maybe_good_first_hand = [[-5, 20], [8, 20], [-8, 30], [10, 30]]
        self.actions_maybe_good_second_hand = [[0.1, 50], [5, 60], [-5, 60], [0.1, 60]]
        self.if_first_hand = None
        self.first_round = True

    def rolling_reset(self):
        self.count = 0
        self.action_mark = None

    def round_reset(self):
        self.count = 0
        self.ball_first = 0
        self.action_mark = None
        self.actions_maybe_good_first_hand = [[-5, 20], [8, 20], [-8, 30], [10, 30]]
        # self.actions_maybe_good_second_hand = [[0.1, 50], [5, 70], [-5, 70], [0.1, 70]]
        self.actions_maybe_good_second_hand = [[0.1, 50], [5, 60], [-5, 60], [0.1, 60]]

        self.if_first_hand = not self.if_first_hand
        
    def anaylse(self, observation):
        if isinstance(observation['obs'], list):
            
            ball_left_current = observation['throws left'][observation['controlled_player_index']]
            
            if  self.count != 0 and ball_left_current != self.ball_left:
                self.rolling_reset()
                self.ball_first += 1
            
            if self.ball_first == 0:
                self.contrl_player_idx = observation['controlled_player_index']
                self.contrl_oppo_player_idx = 1 - self.contrl_player_idx
                self.if_first_hand = True if observation['throws left'][self.contrl_oppo_player_idx] == 4 else False

            if ball_left_current == 3 and self.ball_first == 4:
                self.round_reset()

            self.obs = observation
            self.ball_left = self.obs['throws left'][observation['controlled_player_index']]
            
            # purple 0 - control_idx 0
            # green 1 - control_idx 1


            # ===================================== 后手策略 ========================================
            if not self.if_first_hand:
                angle = 0
                force = 100
                if self.count == self.mark:
                    idxs = np.where(self.obs['obs'][0] == COLOR_TO_IDX[team_id[self.contrl_oppo_player_idx]]) # TODO
                    if len(idxs[0]) > 0:
                        xs, ys = idxs[0], idxs[1]
                        oppo_ball_already = 4 - self.obs['throws left'][self.contrl_oppo_player_idx] # TODO
                        avg_x_poss, avg_y_poss = None, None
                        if oppo_ball_already > 1:
                            last_idx = 0
                            dist = 50
                            for i in range(len(xs)):
                                if i == (len(xs)-1) or abs(xs[i] - xs[i+1]) > 2 or abs(ys[i] - ys[i+1]):
                                    avg_x, avg_y = np.mean(xs[last_idx:i+1]), np.mean(ys[last_idx:i+1])
                                    last_idx = i+1
                                    if (avg_x - 8.5) ** 2 + (avg_y - 14.5) ** 2 < dist:
                                        dist = (avg_x - 8.5) ** 2 + (avg_y - 14.5) ** 2
                                        avg_x_poss, avg_y_poss = avg_x, avg_y
                        else:
                            avg_x, avg_y = np.mean(xs), np.mean(ys)
                            if (avg_x - 8.5) ** 2 + (avg_y - 14.5) ** 2 < 50:
                                avg_x_poss, avg_y_poss = avg_x, avg_y
                        if not avg_x_poss:
                            angle = 0
                        else:
                            angle = -180*math.atan((15 - avg_y_poss)/ (30 - avg_x_poss))/math.pi
                        if abs(angle) > 30:
                            angle = 30 if angle > 0 else -30

                    if angle == 0:
                        action = self.actions_maybe_good_second_hand.pop()
                        angle, force = action[0], action[1]
                    else:
                        if abs(angle) > 20:
                            force = 100
                    self.action_mark = (angle, force)

                elif self.count > self.mark:
                    action, force = 0, self.action_mark[1]
        
            else: 

                # ===================================== 先手策略 ========================================
                angle = 0
                force = 100
                if self.count == self.mark:
                    idxs = np.where(self.obs['obs'][0] == COLOR_TO_IDX[team_id[self.contrl_oppo_player_idx]])
                    if len(idxs[0]) > 0:
                        xs, ys = idxs[0], idxs[1]
                        oppo_ball_already = 4 - self.obs['throws left'][self.contrl_oppo_player_idx
                        ] # TODO
                        avg_x_poss, avg_y_poss = None, None
                        if oppo_ball_already > 1:
                            last_idx = 0
                            dist = 100
                            for i in range(len(xs)):
                                if i == (len(xs)-1) or abs(xs[i] - xs[i+1]) > 2 or abs(ys[i] - ys[i+1]):
                                    avg_x, avg_y = np.mean(xs[last_idx:i+1]), np.mean(ys[last_idx:i+1])
                                    last_idx = i+1
                                    if (avg_x - 8.5) ** 2 + (avg_y - 14.5) ** 2 < dist:
                                        dist = (avg_x - 8.5) ** 2 + (avg_y - 14.5) ** 2
                                        avg_x_poss, avg_y_poss = avg_x, avg_y
                        else:
                            avg_x, avg_y = np.mean(xs), np.mean(ys)
                            if (avg_x - 8.5) ** 2 + (avg_y - 14.5) ** 2 < 100:
                                avg_x_poss, avg_y_poss = avg_x, avg_y
                        if not avg_x_poss:
                            angle = 0
                        else:
                            angle = -180*math.atan((15 - avg_y_poss)/ (30 - avg_x_poss))/math.pi
                        if abs(angle) > 30:
                            angle = 30 if angle > 0 else -30

                    if angle == 0:
                        action = self.actions_maybe_good_first_hand.pop()
                        angle, force = action[0], action[1]
                    else:
                        if abs(angle) > 20:
                            force = 100
                    self.action_mark = (angle, force)

                elif self.count > self.mark:
                    angle, force = 0, self.action_mark[1]
            return angle, force
        
        else:
            return None

    def step(self, observation):
        actions = self.anaylse(observation)
        if actions:
            angle, force = actions[0], actions[1]
            if self.count <= 3:
                action = [160, 0]
            elif 3 < self.count <= 4:
                action = [-100, 0]
            elif self.count > self.mark:
                action = [force, 0]
            elif self.count == self.mark:
                action = [force, angle]
            else:
                action = [0, 0]
            self.count += 1
            print('force', action[0])
            return action
        else:
            return None


rule = rule_agent()


def my_controller(observation, action_space, is_act_continuous=False):
    # inference
    action_ctrl = rule.step(observation)
    if action_ctrl:
        agent_action = [[action_ctrl[0]], [action_ctrl[1]]]  # wrapping up the action
    else:
        agent_action = [[0], [0]]
    return agent_action
