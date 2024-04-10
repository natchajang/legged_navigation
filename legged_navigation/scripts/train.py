# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# append root path of project to find all module in project
import sys
sys.path.append("/home/natcha/github/legged_navigation")

import numpy as np
import os
import inspect
from datetime import datetime

import isaacgym
from legged_navigation.envs import *
from legged_navigation.utils import get_args, task_registry, helpers
import torch

from isaacgym import gymutil

# train function
def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    if args.command_viz and isinstance(env, AnymalNav):
        env.commands_viz = True
    
    if args.camera_viz and isinstance(env, AnymalNav):
        env.camera_image_viz = True
        
    # print env and policy info form configuration
    print("task: {} run name: {}------------------------".format(args.task, train_cfg.runner.run_name))
    print("num_envs : {} mesh_type : {}".format(env.num_envs, env.cfg.terrain.mesh_type))
    print("State space\n num_observations: {}\n measure height: {}\n".format(env_cfg.env.num_observations, env_cfg.terrain.measure_heights))
    # if isinstance(env_cfg, AnymalCNavCfg): print("Reward\n function type:{}\n".format(env_cfg.rewards.reward_tracking))
    # if isinstance(env_cfg, AnymalCNavCfg): print("Camera Sensor\n Active: {}\n Image Type: {}\n".format(env_cfg.camera.active, env_cfg.camera.image_type))
    print("Terrain\n mesh type: {}\n ".format(env_cfg.terrain.mesh_type))
    print("Techniuqe\n terrain cur: {}\n command cur: {}\n positive rew: {}\n".format(env_cfg.terrain.curriculum,
                                                                                    env_cfg.commands.curriculum,
                                                                                    env_cfg.rewards.only_positive_rewards))
    print("Resume : {}\n".format(train_cfg.runner.resume))
    print("----------------------------------------------")

    # train ppo
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    
    # save obj for play next time
    helpers.save_env_cfg(env_cfg,  train_cfg, ppo_runner)
    # save log of reward if it has logger attribute
    if 'logger' in dir(env):
        print(type(env.logger))
        helpers.save_log(env.logger.state_log, ppo_runner.log_dir, 'training_log.pkl')

if __name__ == '__main__':
    args = get_args()
    train(args)