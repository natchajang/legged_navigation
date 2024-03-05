# append root path of project to find all module in project
import sys
sys.path.append("/home/natcha/github/legged_navigation")

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_navigation.envs import *
from legged_navigation.utils import get_args, task_registry
import torch

from isaacgym import gymutil

# train function
def train(args):
    # create object of task (e.g., AnymalEdit) after register in __init__.py
    env, env_cfg = task_registry.make_env(name=args.task, args=args) # create object of task (e.g., AnymalEdit) after register in __init__.py
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    # print(task_registry.task_classes['anymal_c_flat'].cfg.control.use_actuator_network)
    # env, env_cfg = task_registry.make_env(name="anymal_c_rough")
    # print(dir(LeggedRobotCfg))
    # print(task_registry.env_cfgs['anymal_c_flat'])
    args = get_args()
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print(env.reward_names)
    print(env.reward_scales)
    print(env.num_envs)
    print(isinstance(env, AnymalNav))