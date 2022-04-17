# -*- coding:utf-8  -*-
# Time  : 2022/3/27 下午: 16:00
# Author: Yutongamber


import numpy as np
import math
import random
import json
import os

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

# from env core
tau = 0.1
gamma = 0.98

COLOR_TO_IDX = {
    'purple': 5,
    'green': 1
}


def cal_move_log_json():
    # while True:
    xs_list = []
    xs_dict = {}
    for a1 in range(200, 0, -1):
        for a2 in range(-100, 1):
            for t1 in range(1, 21):
                for t2 in range(1, 21):
                    x, v = 0, 0
                    for _ in range(t1):
                        x_delta = v * 0.1
                        x = x + x_delta
                        v_delta = a1 * 0.1
                        v = v * gamma + v_delta
                    for _ in range(t2):
                        x_delta = v * 0.1
                        x = x + x_delta
                        v_delta = a2 * 0.1
                        v = v * gamma + v_delta
                    if abs(v) <= 10 and x > 0 and x < 150:
                        print("good")
                        print("a1, a2, t1, t2, x, v: ", a1, a2, t1, t2, x, v)
                        if round(x) not in xs_list:
                            xs_list.append(round(x))
                            xs_dict[round(x)] = {"a1": a1, "a2": a2, "t1": t1, "t2": t2, "v": v}
    xs_list.sort()
    print("check: ", xs_list)
    with open('log' + '.json', 'w') as f:
        f.write(json.dumps(xs_dict))

def test_move_formular(a1, a2, t1, t2):
    x, v = 0, 0
    for _ in range(t1):
        x_delta = v * 0.1
        x = x + x_delta
        v_delta = a1 * 0.1
        v = v * gamma + v_delta
    for _ in range(t2):
        x_delta = v * 0.1
        x = x + x_delta
        v_delta = a2 * 0.1
        v = v * gamma + v_delta
    print("a1, a2, t1, t2: ", a1, a2, t1, t2)
    print("result x: ", x)
    print("result v: ", v)
    return x


def load_record(path):
    file = open(path, "rb")
    filejson = json.load(file)
    return filejson

base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "log.json")
file = load_record(file_path)

def check_log_json():
    for i in range(0, 151):
        if str(i) not in file:
            assert "please check {}".format(i)
        else:
            print(i)

class rule_agent:
    def __init__(self):
        self.count = 0 # 这个是记录局内的步数的
        self.obs = None
        self.ball_left = None
        self.ball_first = 0
        self.mark_2 =  42 # hyper
        self.mark_1 = 30
        self.contrl_player_idx = None
        self.ball_oppo_pos = []
        self.action_mark = None
        self.actions_maybe_good_first_hand = [[-5, 20], [8, 20], [-8, 30], [10, 30]]
        self.actions_maybe_good_second_hand = [[0.1, 50], [5, 60], [-5, 60], [0.1, 60]]
        self.if_first_hand = None
        self.first_round = True
        self.first_ball_FirstHand=False
        self.last_ball_FirstHand=False

        # 速度; 位置; 加速度
        self.v = 0
        self.pos = [300, 150]
        self.a = None
        self.key_pos = None
        self.move = None

        self.turn_cnt_1 = 0
        self.turn_cnt_2 = 0
        self.move_cnt = 0
        self.move_cnt_org = 0

    def rolling_reset(self):
        self.count = 0
        self.action_mark = None

        self.v = 0
        self.pos = [300, 150]
        self.a = None
        self.key_pos = None
        self.move = None

        self.turn_cnt_1 = 0
        self.turn_cnt_2 = 0
        self.move_cnt = 0
        self.move_cnt_org = 0

    def round_reset(self):
        self.count = 0
        self.ball_first = 0
        self.action_mark = None
        self.actions_maybe_good_first_hand = [[-5, 80], [8, 80], [-8, 80], [10, 80]] #default:[[-5, 20], [8, 20], [-8, 30], [10, 30]]
        # self.actions_maybe_good_second_hand = [[0.1, 50], [5, 70], [-5, 70], [0.1, 70]]
        self.actions_maybe_good_second_hand = [[0.1, 50], [5, 60], [-5, 60], [0.1, 60]]

        self.if_first_hand = not self.if_first_hand
    
    def if_oppo_in_circle(self):
        check = False
        idxs = np.where(self.obs['obs'][0] == COLOR_TO_IDX[team_id[self.contrl_oppo_player_idx]])        
        
        if len(idxs[0]) > 0:
            xs, ys = idxs[0], idxs[1]
            oppo_ball_already = 4 - self.obs['throws left'][self.contrl_oppo_player_idx] # TODO
            last_idx = 0
            dist = 50
            for i in range(len(xs)):
                if i == (len(xs)-1) or abs(xs[i] - xs[i+1]) > 2 or abs(ys[i] - ys[i+1]):
                    avg_x, avg_y = np.mean(xs[last_idx:i+1]), np.mean(ys[last_idx:i+1])
                    last_idx = i+1
                    if (avg_x - 8.5) ** 2 + (avg_y - 14.5) ** 2 < dist:
                        check = True
                        break
            # TODO: CHECK 只有一个球的时候
            # else:
            #     avg_x, avg_y = np.mean(xs), np.mean(ys)
            #     if (avg_x - 8.5) ** 2 + (avg_y - 14.5) ** 2 < 100:
            #         avg_x_poss, avg_y_poss = avg_x, avg_y
        return check

    def calculate_velocity(self):
        self.v_delta = self.a * tau
        self.v = self.v * gamma + self.v_delta
    
    def calculate_pos(self):
        self.pos[1] += self.v * tau
    
    def get_force_v_0(self):
        force = (-self.v * gamma) * 10
        return force
    
    def save_obs(self, obs):
        self.obs = obs
        self.key_pos = self.pos

    def cal_move(self):
        # 计算水平移动
        idxs = np.where(self.obs['obs'][0] == COLOR_TO_IDX[team_id[self.contrl_oppo_player_idx]]) 
        if len(idxs[0]) > 0:
            # TODO: 有几个球的情况
            move = True
            xs, ys = idxs[0], idxs[1]
            avg_x, avg_y = np.mean(xs), np.mean(ys)
            print("check avg_x: ", avg_x)
            print("check avg_y: ", avg_y)
            self.move = avg_y / 30 * 300 - 150
            # TODO: 如果没有对方球 会报错
            self.move = abs(self.move)
            self.move_cnt_org = file[str(round(self.move))]["t1"] + file[str(round(self.move))]["t2"]
            # self.move_a1 = file[round(self.move)]["a1"]
            # self.move_a2 = file[round(self.move)]["a2"]
    
    def make_turn_move_shoot(self):
        if self.move:
            left = True if self.move > 0 else False
            move = file[str(round(self.move))]

            # turn 
            if self.move_cnt_org > 0 and self.turn_cnt_1 < 3:
                force = 0
                if left:
                    if self.turn_cnt_1 == 0: 
                        angle = 30
                    elif self.turn_cnt_1 == 1:
                        angle = 30
                    else:
                        angle = 30
                else:
                    if self.turn_cnt_1 == 0: 
                        angle = -30
                    elif self.turn_cnt_1 == 1:
                        angle = -30
                    else:
                        angle = -30
                self.turn_cnt_1 += 1
            
                return angle, force
            else: 
                pass
                # TODO: 如果不需要转
            
            # move 
            a1 = move["a1"]
            a2 = move["a2"]
            t1 = move["t1"]
            t2 = move["t2"]
            if self.move_cnt_org > 0 and self.move_cnt < t1 + t2:
                angle = 0
                if t1 - self.move_cnt > 0:
                    force = a1
                elif t1 - self.move_cnt <= 0:
                    force = a2
                self.move_cnt += 1
                return 0, force
            
            if self.move_cnt == t1 + t2:
                force = (-self.v * gamma) * 10
                self.move_cnt += 1
                return 0, force
            
            if self.move_cnt_org > 0 and self.turn_cnt_2 < 3:
                force = 0
                if left:
                    if self.turn_cnt_2 == 0: 
                        angle = -30
                    elif self.turn_cnt_2 == 1:
                        angle = -30
                    else:
                        angle = -30
                else:
                    if self.turn_cnt_2 == 0: 
                        angle = 30
                    elif self.turn_cnt_2 == 1:
                        angle = 30
                    else:
                        angle = 30
                self.turn_cnt_2 += 1
                return angle, force
            
            return 0, 200
        else:
            return 0, 50
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
            # print('======================================')
            # print(self.obs['obs'][0])

            # ===================================== 后手策略 ========================================
            if not self.if_first_hand:
                angle = 0
                force = 200
                if self.count == self.mark_2:
                    idxs = np.where(self.obs['obs'][0] == COLOR_TO_IDX[team_id[self.contrl_oppo_player_idx]]) # TODO
                    if len(idxs[0]) > 0:
                        xs, ys = idxs[0], idxs[1]
                        oppo_ball_already = 4 - self.obs['throws left'][self.contrl_oppo_player_idx] # TODO
                        avg_x_poss, avg_y_poss = None, None
                        if oppo_ball_already > 1:
                            last_idx = 0
                            dist = 100000
                            for i in range(len(xs)):
                                if i == (len(xs)-1) or abs(xs[i] - xs[i+1]) > 2 or abs(ys[i] - ys[i+1]):
                                    avg_x, avg_y = np.mean(xs[last_idx:i+1]), np.mean(ys[last_idx:i+1])
                                    last_idx = i+1
                                    if (avg_x - 8.5) ** 2 + (avg_y - 14.5) ** 2 < dist:
                                        dist = (avg_x - 8.5) ** 2 + (avg_y - 14.5) ** 2
                                        avg_x_poss, avg_y_poss = avg_x, avg_y
                        else:
                            avg_x, avg_y = np.mean(xs), np.mean(ys)
                            if (avg_x - 8.5) ** 2 + (avg_y - 14.5) ** 2 < 100000:
                                avg_x_poss, avg_y_poss = avg_x, avg_y
                        if not avg_x_poss and not avg_y_poss:
                            angle = 0
                        else:
                            angle = -180*math.atan((15 - avg_y_poss)/ (30 - avg_x_poss))/math.pi
                        if abs(angle) > 30:
                            angle = 30 if angle > 0 else -30

                    if angle == 0:
                        action = self.actions_maybe_good_second_hand.pop()
                        angle, force = action[0], action[1] # default: 
                    # else:
                        # if abs(angle) > 20:
                            # force = 100
                    self.action_mark = (angle, force)

                elif self.count > self.mark_2:
                    action, force = 0, self.action_mark[1]
        
            else: 
                # 似乎最后一个球，没必要放在中心，可以放在圈外
                # ===================================== 先手策略 ========================================
                
                oppo_ball_already1 = 4 - self.obs['throws left'][self.contrl_oppo_player_idx]
                if oppo_ball_already1==0:
                    self.first_ball_FirstHand=True
                    angle = 0 # 前30步的动作
                    force = 150
                else:
                    angle = 0 # 前30步的动作
                    force = 150
                    self.first_ball_FirstHand=False
                if oppo_ball_already1==3: # 我方最后一个球
                    self.last_ball_FirstHand=True
                else:
                    self.last_ball_FirstHand=False
                if self.count == self.mark_1:
                    idxs = np.where(self.obs['obs'][0] == COLOR_TO_IDX[team_id[self.contrl_oppo_player_idx]])
                   
                    # print(idxs) # (array([3, 3, 3, 4, 4, 4, 5]), array([5, 6, 7, 5, 6, 7, 6])) 现在场上有两个球
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
                            force = 200 #default: 100
                    oppo_ball_already1 = 4 - self.obs['throws left'][self.contrl_oppo_player_idx]
                    
                                          
                    if oppo_ball_already1==0: # 我方第一个球
                        angle=0
                        force=50  
                    self.action_mark = (angle, force)

                elif self.count > self.mark_1: 
                    angle, force = 0, self.action_mark[1] # 这个作用是保证小球的轨迹保持第30步的决策，让其按照30步的决策
            return angle, force
        
        else:
            return None

    def step(self, observation):
        actions = self.anaylse(observation)
        print("=============================")
        print("count: ", self.count)
        self.mark_2 = 41
        if not self.if_first_hand: # 我方是后手
            if actions:
                angle, force = actions[0], actions[1]
                if self.count < 8:
                    action = [200, 0]
                if self.count >= 8 and self.count < 18:
                    action = [-100, 0]
                if self.count >= 18 and self.count < 35: 
                    action = [-100, 0]
                    if self.count == 20:
                        self.save_obs(observation) # 保存obs; 并记录agent pos
                        self.cal_move()
                if self.count >= 35 and self.count < 40:
                    action = [200, 0]
                if self.count == 40:
                    force = self.get_force_v_0()
                    action = [force, 0]
                if self.count > 40:
                    angle, force = self.make_turn_move_shoot()
                    action = [force, angle]
                # elif self.count >= self.mark_2:
                #     action = [force, 0]
                # elif self.count == self.mark_2:
                #     action = [force, angle]
                self.count += 1
                self.a = action[0]
                # 需要先计算位置,再计算速度
                self.calculate_pos()
                self.calculate_velocity()
                print("check v in agent: ", self.v)
                print("check pos in agent: ", self.pos)
                print("check x in reality: ", 0.5 * self.a * (self.count*0.1)**2)
                return action
            else:
                return None
        else:   # 我方是先手
            if actions:
                angle, force = actions[0], actions[1]
                if self.first_ball_FirstHand==True: # 第一个球这么放，后面的球加大力度
                    if self.count <= 0:
                        action = [20, 30]
                    elif self.count > self.mark_1:
                        action = [5, 0]
                    elif self.count == self.mark_1:
                        action = [force, angle]
                    else:
                        action = [20, 0]
                elif self.last_ball_FirstHand==True: # NOTE:最后一个球，这个球的力度再减少一点
                    if self.count <= 0:
                        action = [15, 0]
                    elif self.count > self.mark_1:
                        action = [4, 0]
                    elif self.count == self.mark_1:
                        action = [force, 0]
                    else:
                        action = [10, 0]
                else:
                    if force==200:
                        force=force
                    else:
                        force=200
                    if self.count <= 0:
                        action = [200, 0]
                    # elif 3 < self.count <= 4:
                    #     action = [-100, 0]
                    elif self.count > self.mark_1:
                        action = [force, 0]
                    elif self.count == self.mark_1:
                        action = [force, angle]
                    else:
                        action = [100, 0]
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
    print("agent action: ", agent_action)
    return agent_action

if __name__ == "__main__":
    check_log_json()
    # test_move_formular(a1=200, a2=0, t1=2, t2=0)
    # test_move_formular(a1=200, a2=-100, t1=8, t2=10)
    # test_move_formular(a1=192, a2=-46, t1=4, t2=16)
    # cal_move_log_json()
    
