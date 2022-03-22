from rl_trainer.algo.random import random_agent
from agents.rule_2.submission import *
from rl_trainer.algo.ppo import PPO
from rl_trainer.log_path import *
from env.chooseenv import make
from collections import deque, namedtuple
import argparse
import datetime
import math
import random

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import wandb
from pathlib import Path
import sys
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)


parser = argparse.ArgumentParser()
parser.add_argument('--game_name', default="olympics-curling", type=str)
parser.add_argument('--algo', default="ppo", type=str, help="ppo")
parser.add_argument('--controlled_player', default=1,
                    help="0(agent purple) or 1(agent green)")
parser.add_argument('--max_episodes', default=5000, type=int)  # NOTE:原先是1500
parser.add_argument('--opponent', default="run1",
                    help="random or run11")  # NOTE:默认值是random
parser.add_argument('--opponent_load_episode', default=1500)  # NOTE:默认值是None

parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--render', action='store_true')

parser.add_argument("--save_interval", default=100, type=int)
parser.add_argument("--model_episode", default=0, type=int)

parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
parser.add_argument("--load_run", default=1, type=int)
parser.add_argument("--load_episode", default=1500, type=int)
parser.add_argument("--ENTROPY_COEFF", default=0.1, type=float)
parser.add_argument("--user_name", type=str, default='lzw123',
                    help="[for wandb usage], to specify user's name for simply collecting training data.")
parser.add_argument("--use_wandb", action='store_false', default=False,
                    help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}  # dicretise action space


def compute_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx**2 + dy**2)


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def main(args):

    run_dirs = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
        0] + "/results") / args.game_name / args.algo / args.game_name
    if not run_dirs.exists():
        os.makedirs(str(run_dirs))

    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    if args.use_wandb == True:
        run = wandb.init(config=args,
                         project=args.game_name,
                         entity=args.user_name,
                         name=str(args.algo) + "_" +
                         str(args.game_name) +
                         "_seed" + str(args.seed),
                         dir=str(run_dirs),
                         job_type="training",
                         reinit=True)
    print("==algo: ", args.algo)
    print(f'device: {device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')
    RENDER = args.render

    env = make(args.game_name)  # build curling env

    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')

    ctrl_agent_index = int(args.controlled_player)

    print(f'Agent control by the actor: {ctrl_agent_index}')

    # ctrl_agent_num = 1

    width = env.env_core.view_setting['width'] + \
        2*env.env_core.view_setting['edge']
    height = env.env_core.view_setting['height'] + \
        2*env.env_core.view_setting['edge']
    print(f'Game board width: {width}')
    print(f'Game board height: {height}')

    act_dim = env.action_dim
    obs_dim = 30*30
    print(f'action dimension: {act_dim}')
    print(f'observation dimension: {obs_dim}')

    torch.manual_seed(args.seed)
    # 定义保存路径

    if not args.load_model:
        writer = SummaryWriter(os.path.join(str(log_dir), "{}_{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.algo)))
        save_config(args, log_dir)

    record_win = deque(maxlen=100)
    record_win_op = deque(maxlen=100)

    if args.load_model:  # build our model
        model = PPO()
        load_dir = os.path.join(os.path.dirname(
            run_dir), "run" + str(args.load_run))
        model.load(load_dir, episode=args.load_episode)
    else:
        # print('I here')如果是到了这里的话，那么是在run_dir保存自己训练好的模型
        model = PPO(run_dir)
        Transition = namedtuple(
            'Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])

    RuleAgent = rule_agent()
    RandomAgent = random_agent()
    PPOAgent = PPO()
    opponent_model = [RandomAgent, RuleAgent, PPOAgent]
    choose_opponent_prob = [0.3,0.4 ,0.3 ]

    '''
    if args.opponent == 'random':
        opponent_agent = random_agent()
    else:
        opponent_agent = PPO()
        opponent_load_dir = os.path.join(
            os.path.dirname(run_dir), args.opponent)
        assert os.path.exists(opponent_load_dir), print(
            'the opponent model path is incorrect!')
        opponent_agent.load(opponent_load_dir,
                            episode=args.opponent_load_episode)
    '''

    episode = 0
    train_count = 0

    while episode < args.max_episodes:
        # [{'obs':[25,25], "control_player_index": 0}, {'obs':[25,25], "control_player_index": 1}]
        state = env.reset()
        all_observes=env.all_observes
        if RENDER:
            env.env_core.render()
        obs_ctrl_agent = np.array(
            state[ctrl_agent_index]['obs']).flatten()  # [30*30]
        obs_oppo_agent = np.array(
            state[1-ctrl_agent_index]['obs']).flatten()  # [30*30]

        episode += 1
        step = 0
        Gt = 0

        opponent_agent = random_pick(opponent_model, choose_opponent_prob)
        opponent_agent=opponent_model[1]
        if opponent_agent == opponent_model[2]:
            opponent_load_dir = os.path.join(
                os.path.dirname(run_dir), args.opponent)
            assert os.path.exists(opponent_load_dir), print(
                'the opponent model path is incorrect!')
            opponent_agent.load(opponent_load_dir,
                                episode=args.opponent_load_episode)

        while True:

            ################################# collect opponent action #############################
            # oppo_action_raw, _ = opponent_agent.select_action(
            #     obs_oppo_agent, False)
            '''
            if args.opponent != 'random':
                oppo_action = actions_map[oppo_action_raw]
                action_opponent = [[oppo_action[0]], [oppo_action[1]]]
            else:
                action_opponent = oppo_action_raw
                # action_opponent = [[50], [0]]
            '''
            if opponent_agent == opponent_model[0]:
                oppo_action_raw, _ = opponent_agent.select_action(
                    obs_oppo_agent, False)
                action_opponent = oppo_action_raw
            elif opponent_agent == opponent_model[1]:  # rule agent
                action_opponent=my_controller_fixed(all_observes[1-ctrl_agent_index])
                
            else:
                oppo_action_raw, _ = opponent_agent.select_action(
                    obs_oppo_agent, False)
                oppo_action = actions_map[oppo_action_raw]
                action_opponent = [[oppo_action[0]], [oppo_action[1]]]

            ################################# collect our action ################################
            action_ctrl_raw, action_prob = model.select_action(
                obs_ctrl_agent, False if args.load_model else True)
            # inference
            action_ctrl = actions_map[action_ctrl_raw]
            # wrapping up the action
            action_ctrl = [[action_ctrl[0]], [action_ctrl[1]]]
           
            # print(action_ctrl) # 输出的形式是 [[20], [-18]]
            action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [
                action_ctrl, action_opponent]

            ################################# env rollout ##########################################
            next_state, reward, done, _, info = env.step(action)

            next_obs_ctrl_agent = next_state[ctrl_agent_index]['obs']
            next_obs_oppo_agent = next_state[1-ctrl_agent_index]['obs']

            step += 1

            if not done:
                post_reward = [-1., -1.]
            else:
                if reward[0] != reward[1]:
                    post_reward = [reward[0]-100, reward[1]] if reward[0] < reward[1] else [
                        reward[0], reward[1]-100]  # 证明了这个场景是一个零和博弈
                else:
                    post_reward = [-1., -1.]

            if not args.load_model and env.env_core.current_team == ctrl_agent_index:
                post_reward[ctrl_agent_index] = - \
                    compute_distance([300, 500], env.env_core.agent_pos[-1])
                trans = Transition(obs_ctrl_agent, action_ctrl_raw, action_prob, post_reward[ctrl_agent_index],
                                   next_obs_ctrl_agent, done)
                model.store_transition(trans)

            obs_oppo_agent = np.array(next_obs_oppo_agent).flatten()
            obs_ctrl_agent = np.array(next_obs_ctrl_agent).flatten()
            all_observes=env.all_observes
            if RENDER:
                env.env_core.render()
            Gt += reward[ctrl_agent_index] if done else 0

            if done:
                win_is = 1 if reward[ctrl_agent_index] > reward[1 -
                                                                ctrl_agent_index] else 0
                win_is_op = 1 if reward[ctrl_agent_index] < reward[1 -
                                                                   ctrl_agent_index] else 0
                record_win.append(win_is)
                record_win_op.append(win_is_op)
                print("Episode: ", episode, "controlled agent: ", ctrl_agent_index, "; Episode Return: ", Gt,
                      "; win rate(controlled & opponent): ", '%.2f' % (
                          sum(record_win)/len(record_win)),
                      '%.2f' % (sum(record_win_op)/len(record_win_op)), '; Trained episode:', train_count)
                if args.use_wandb == True:
                    wandb.log({'Episode Return': Gt})
                    wandb.log({'win rate(controlled & opponent)': (
                        sum(record_win)/len(record_win))})

                if not args.load_model:
                    # 当buffer大小超过设定值的时候才进行训练
                    if args.algo == 'ppo' and len(model.buffer) >= model.batch_size:
                        if win_is == 1:
                            model.update(episode)
                            train_count += 1
                        else:
                            model.clear_buffer()

                    writer.add_scalar('training Gt', Gt, episode)

                break
        if episode % args.save_interval == 0 and not args.load_model:
            model.save(run_dir, episode)
    if args.use_wandb == True:
        run.finish()


if __name__ == '__main__':
    args = parser.parse_args()

    # args.controlled_player = 0
    # args.opponent = 'random'
    # args.max_episodes = 5000
    # args.save_interval = 200
    # args.opponent_load_episode=1500
    # args.render = True
    #
    # args.load_model = True
    # args.load_run = 2
    # args.load_episode = 1500

    args.render = True
    main(args)
    # print(args.game_name)
