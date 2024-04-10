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
    #write over config from saved file
    helpers.rideover_cfg(load_path, env_cfg, train_cfg)

    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50) # limit number of visualization environments
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.vertical_scale = 0.01
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    env_cfg.env.episode_length_s = 20
    env_cfg.commands.resampling_time = 3
    
    #set viewer pos and lookat
    env_cfg.viewer.pos = [16, 0, 5] #[11, 5, 2]
    env_cfg.viewer.lookat = [8, 8,0]#[8, 8, 0]
    
    move_came = False
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([0.3, 0, 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # set needed variable
    logger = Logger(env.dt)
    number_iter = 1
    episode_length = 1000
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    
    # #Add logger for save and analysis
    # logger_save = Logger(env.dt)
    # logger_save.log_state('dt', env.dt)
    # logger_save.log_state('num_samples', number_iter*episode_length)
    
    # change command if set desired command
    if args.command_set != None:
        env.command_set_flag =True
        command_set = helpers.comand_set_parse(args.command_set)
        list_timestamp = []
        list_commands = []
        command_idx = 0
        episode_length = 0
        for c in command_set:
            # add time stamp to switch commands
            episode_length += int(c[-1]/env.dt)
            list_timestamp.append(episode_length)
            c = [e*math.pi/180 if i>2 and i<6 else e for i,e in enumerate(c)][:-1]
            c[0] = c[0]/100
            c[1] = c[1]/100
            c[2] = c[2]/100
            command = torch.tensor(c, device=env.device, requires_grad=False)
            command = command.repeat(env_cfg.env.num_envs, 1)
            list_commands.append(command)
        env.cfg.commands.resampling_time = 2*int(episode_length/env.dt)
        env.command_own = list_commands[command_idx]
        
    #Add logger for save and analysis
    logger_save = Logger(env.dt)
    logger_save.log_state('dt', env.dt)
    logger_save.log_state('num_samples', number_iter*episode_length)
    
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
    print("Load run\n run: {}\n model: {}".format(train_cfg.runner.load_run, train_cfg.runner.checkpoint))
    print("----------------------------------------------")
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
        
    first_state = env.root_states[:, :2].clone().detach()
    desired_distance = torch.norm(env.commands[:, :2], dim=1)*20*0.8
    for i in range(number_iter*int(episode_length)):

        if args.command_set != None:
            if i==list_timestamp[command_idx]:
                command_idx+=1
                env.command_own = list_commands[command_idx]
                
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
        
        # logger_save.log_states(
        #         {
        #             'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
        #             'dof_pos': env.dof_pos[robot_index, joint_index].item(),
        #             'dof_vel': env.dof_vel[robot_index, joint_index].item(),
        #             'dof_torque': env.torques[robot_index, joint_index].item(),
        #             'command_lin_x': env.commands[robot_index, 0].item(),
        #             'command_lin_y': env.commands[robot_index, 1].item(),
        #             'command_height': env.commands[robot_index, 2].item(),
        #             'command_roll': env.commands[robot_index, 3].item(),
        #             'command_pitch': env.commands[robot_index, 4].item(),
        #             'command_yaw': env.commands[robot_index, 5].item(),
        #             'base_roll' : env.base_euler[robot_index, 0].item(),
        #             'base_pitch' : env.base_euler[robot_index, 1].item(),
        #             'base_yaw' : env.base_euler[robot_index, 2].item(),
        #             'base_pos_x' : env.base_mean_height[robot_index].item(),    
        #             'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
        #             'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
        #             'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
        #             'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
        #             'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
        #             'distance_x' : env.root_states[robot_index, 0].item(),
        #             'distance_y' : env.root_states[robot_index, 1].item()
        #         }
            # )
    # desired_distance = torch.norm(env.commands[:, :2], dim=1)*20*0.5
    # print("Command: ", torch.norm(env.commands[:, :2], dim=1))
    walk_distance = torch.norm(env.root_states[:, :2]-first_state, dim=1)
    print("Desired Distance: ", desired_distance)
    print("Walk Distance: ", walk_distance)
    reach_half = torch.where(walk_distance> desired_distance, 1, 0)
    print("Reach half distance", reach_half)
    print("Sum", torch.sum(reach_half))
    
    helpers.save_log(logger_save.state_log, load_path, 'state_log.pkl')

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
    