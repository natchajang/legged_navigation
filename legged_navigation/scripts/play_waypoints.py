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

from legged_navigation import LEGGED_GYM_ROOT_DIR
import os
import math
import importlib
import inspect
import time
import isaacgym
from isaacgym import gymutil
import legged_navigation
from legged_navigation import envs
from legged_navigation import utils
from legged_navigation.envs import * 
from legged_navigation.utils import  get_args, export_policy_as_jit, Logger, helpers, TaskRegistry

import pickle
import numpy as np
import torch

def play(args):
    load_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.task)
    load_folder = [f for f in os.listdir(load_dir) if args.run_name in f]
    if load_folder == []:
        print("Error; not found that run_name in {} task".format(args.task))
        return
    elif len(load_folder) > 1:
        print("Warning; found that run_name in {} task more than 1 !!!".format(args.task))
    args.load_run = load_folder[0]
    load_path = os.path.join(load_dir, args.load_run)
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # write over config from saved file
    helpers.rideover_cfg(load_path, env_cfg, train_cfg)

    # override some parameters for testing
    env_cfg.commands.reachgoal_resample = True              # change mode to resample when the agent reach the goal
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 5)     # limit number of visualization environments
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.vertical_scale = 0.01
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    time_play_s = 30
    env_cfg.env.episode_length_s = time_play_s
    env_cfg.commands.resampling_time = time_play_s + 1
    env_cfg.commands.ranges.radius[0] = 5
    env_cfg.commands.ranges.radius[1] = 5
    env_cfg.commands.ranges.base_height[0] = 0.6

    # set ways point command (if need)
    # waypoints = [[2, 0, 0.4], [4, 0, 0.3], [6, 0, 0.2], [8, 0, 0.2], [10, 0, 0.4]] # [x, y, z]
    waypoints = [[2, 0, 0.25], [-2, 0, 0.25], [2, 0, 0.25], [-2, 0, 0.25]]
    num_waypoints = 0 * len(waypoints) # if set = 0 command will random
    if num_waypoints != 0:
        env_cfg.env.num_envs = 1
    
    # set viewer pos and lookat
    env_cfg.viewer.pos = [11, 5, 2]     #[11, 5, 2]
    env_cfg.viewer.lookat = [0, 0, 0]    #[8, 8, 0]

    move_came = False
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([0.3, 0, 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if num_waypoints !=0:
        env.num_waypoints = num_waypoints
        env.command_waypoints = torch.tensor(waypoints, device=env.device)
    
    # set log variable for saving and analysis
    logger = Logger(env.dt)
    num_steps = time_play_s / env.dt
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    logger_save = Logger(env.dt)
    logger_save.log_state('dt', env.dt)
    logger_save.log_state('num_samples', num_steps)
    
    # Visulization setting
    # add open debug_viz
    if args.debug_viz:
        env.debug_viz = True
    # add open command visualization
    if args.command_viz and (isinstance(env, AnymalNav) or isinstance(env, AnymalEdit)):
        env.commands_viz = True
        
    # load policy
    train_cfg.runner.resume = True # use for load previous model for continue training or load model for playing by specifi checkpoint and load_run
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # print env and policy info form configuration
    print("replay => run name: {}------------------------".format(train_cfg.runner.run_name))
    print("State space\n num_observations: {}\n measure height: {}\n".format(env_cfg.env.num_observations, env_cfg.terrain.measure_heights))
    # print("Reward\n function type:{}".format(env_cfg.terrain.mesh_type))
    print("Terrain\n mesh type: {}\n ".format(env_cfg.terrain.mesh_type))
    print("Techniuqe\n terrain cur: {}\n command cur: {}\n positive rew: {}".format(env_cfg.terrain.curriculum,
                                                                                    env_cfg.commands.curriculum,
                                                                                    env_cfg.rewards.only_positive_rewards))
    print("Load run\n run: {}\n model: {}\n".format(train_cfg.runner.load_run, train_cfg.runner.checkpoint))
    print("time play: {}".format(time_play_s))
    print("steps play: {}".format(int(num_steps)))
    print("----------------------------------------------")
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    for i in range(int(num_steps)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
 
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if move_came or MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
    
    helpers.save_log(logger_save.state_log, load_path, 'state_log.pkl')

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
    