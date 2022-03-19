# -*- coding:utf-8  -*-
# Time  : 2022/3/27 下午: 16:00
# Author: Yutongamber


import numpy as np
import math


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

class rule_agent:
    def __init__(self):
        self.count = 0
        self.obs = None
        self.obs_key = None
        self.ball_left = None
        self.ball_first = 0
        self.mark = 41 # hyper
        self.contrl_player = None
        self.ball_oppo_pos = []

    def rolling_reset(self):
        self.count = 0

    def get_obs(self, observation):
        ball_left_current = observation['throws left'][observation['controlled_player_index']]

        if self.ball_first == 0:
            self.contrl_player = observation['controlled_player_index']

        if  self.count != 0 and ball_left_current != self.ball_left:
            self.rolling_reset()
            self.ball_first += 1

        if self.ball_first == 1 and self.count == self.mark:
            self.obs_key = observation

        self.obs = observation
        self.ball_left = self.obs['throws left'][observation['controlled_player_index']]


    def analyze(self):
        angle = 0
        force = 200
        if self.count == self.mark:
            idxs = np.where(self.obs['obs'][0] == 1)
            print("check idxs: ", idxs)
            if len(idxs[0]) > 0:
                xs, ys = idxs[0], idxs[1]
                oppo_ball_already = 4 - self.obs['throws left'][1] # TODO
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
                    angle = 0.1
                else:
                    angle = -180*math.atan((15 - avg_y_poss)/ (30 - avg_x_poss))/math.pi
                if abs(angle) > 30:
                    angle = 30 if angle > 0 else -30
            print("check angle: ", angle)

        if angle == 0:
            force = 50
        return angle, force

    def step(self):
        angle, force = self.analyze()
        if self.count <= 5:
            action = [120, 0]
        elif 5 < self.count <= 8:
            action = [-90, 0]
        elif self.count > self.mark:
            action = [force, 0]
        elif self.count == self.mark:
            action = [0, angle]
        else:
            action = [0, 0]
        self.count += 1
        print("count: ", self.count)
        print("==== actions: ", action)
        return action


rule = rule_agent()


def my_controller(observation, action_space, is_act_continuous=False):
    # inference
    rule.get_obs(observation)
    action_ctrl = rule.step()
    agent_action = [[action_ctrl[0]], [action_ctrl[1]]]  # wrapping up the action

    return agent_action
