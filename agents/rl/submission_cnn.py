# -*- coding:utf-8  -*-
# Time  : 2022/1/29 上午10:48
# Author: Yahui Cui
import argparse
import os
import sys

import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


Use_CNN=True
class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64):
        super(Actor, self).__init__()
        
        if Use_CNN==True:
            self.conv1 = nn.Sequential(     # In: 1*2*30*30
            nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            )                            # Out: 1*16*15*15
       
            self.conv2 = nn.Sequential(     # In: 1*16*15*15
                nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(5, stride=2),        # 1*32*6*6
            )
            self.decision_fc = nn.Linear(1152, action_space)
        else:
            self.linear_in = nn.Linear(state_space, hidden_size)
            self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        if Use_CNN==True:
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(-1, 1152)
            action_prob = F.softmax(self.decision_fc(x), dim=1)
        else:
            x = F.relu(self.linear_in(x))
            action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Args:
    action_space = 36
    state_space = 900


ppo_args = Args()
device = 'cpu'


class PPO:
    action_space = ppo_args.action_space
    state_space = ppo_args.state_space

    def __init__(self):
        super(PPO, self).__init__()
        self.args = ppo_args
        self.actor_net = Actor(self.state_space, self.action_space).to(device)
        self.ball_left = None
        self.ball_first = 0
        self.flag = 0  # 判断ego agent是先手还是后手
        self.obs = None
        self.OnlyRunOnce_First = 1
        self.OnlyRunOnce_Mediate = 1
        self.save_ego_ball = 10

    def select_action(self, state, train=False):
        # print(state.shape)
        first_hand_array = np.ones(shape=(1, 30, 30))
        back_hand_array = np.zeros(shape=(1, 30, 30))
        temp = state
        if self.flag == 0:  # ego agent是先手
            obs_ctrl_agent = np.vstack((first_hand_array, temp))
        else: 
            obs_ctrl_agent=np.vstack((back_hand_array,temp))
            
        state = torch.from_numpy(obs_ctrl_agent).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_prob = self.actor_net(state).to(device)
        c = Categorical(action_prob)
        if train:
            action = c.sample()
        else:
            action = torch.argmax(action_prob)
            # action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def load(self, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.dirname(__file__)
        print("base_path: ", base_path)
        model_actor_path = os.path.join(
            base_path, "actor_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')

        if os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            self.actor_net.load_state_dict(actor)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

    def get_obs(self, observation):
        EgoAgent_ThrowLeft = observation['throws left'][observation['controlled_player_index']]
        OppoAgent_ThrowLeft = observation['throws left'][1 -
                                                         observation['controlled_player_index']]
        if (EgoAgent_ThrowLeft < OppoAgent_ThrowLeft) and self.OnlyRunOnce_First == 1:
            self.flag = 0  # ego agent是先手
            self.OnlyRunOnce_First = 0
        
        elif self.OnlyRunOnce_First == 1 and (EgoAgent_ThrowLeft > OppoAgent_ThrowLeft):
            self.flag = 1
            self.OnlyRunOnce_First = 0
        # print('flag is', self.flag)

        if ((self.save_ego_ball-observation['throws left'][observation['controlled_player_index']]) == -3) and self.OnlyRunOnce_Mediate == 1:
            # print('对局切换')
            self.OnlyRunOnce_Mediate = 0
            if self.flag == 1:
                self.flag = 0
            else:
                self.flag = 1


# parser = argparse.ArgumentParser()
# parser.add_argument("--load_episode", default=300, type=int)
# args = parser.parse_args()
model = PPO()
model.load(episode=4600) #episode=4600表示CNN训练的结果 ,这个效果还可以


actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}


def my_controller(observation, action_space, is_act_continuous=False):
    # print(observation)
    # 思路：先判断先手后手，然后判断对手球数量发生变化的时候则切换先后手，明天测试一下类中的变量是不是全局变量
    obs_ctrl_agent = np.array(observation['obs']) #.flatten()
    model.get_obs(observation)
    model.save_ego_ball=observation['throws left'][observation['controlled_player_index']]
    action_ctrl_raw, action_prob = model.select_action(obs_ctrl_agent, False)
    # inference
    action_ctrl = actions_map[action_ctrl_raw]
    # wrapping up the action
    agent_action = [[action_ctrl[0]], [action_ctrl[1]]]

    return agent_action
