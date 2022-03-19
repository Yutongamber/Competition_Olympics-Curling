# -*- coding:utf-8  -*-
# Time  : 2022/3/27 下午: 16:00
# Author: Yutongamber



def my_controller(observation, action_space, is_act_continuous=False):
    # inference
    action_ctrl = [30, 0]
    agent_action = [[action_ctrl[0]], [action_ctrl[1]]]  # wrapping up the action

    return agent_action
