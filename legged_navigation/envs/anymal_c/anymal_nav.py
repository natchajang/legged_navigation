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

from time import time
import numpy as np
import math
import statistics
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from collections import defaultdict, deque
from legged_navigation import LEGGED_GYM_ROOT_DIR
from legged_navigation.envs import LeggedRobot
from legged_navigation.utils.terrain import Terrain
from .navigation.anymal_c_nav_config import AnymalCNavCfg
from legged_navigation.utils.math import quat_apply_yaw, wrap_to_pi, euclidean_distance, transformation_inverse
from legged_navigation.utils import Logger

class AnymalNav(LeggedRobot):
    cfg : AnymalCNavCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # additional attributes
        # flags
        self.commands_viz = False   # for visualizing command
        # command set waypoints
        self.idx_waypoints = torch.zeros(self.num_envs, dtype=int, device=self.device)
        self.num_waypoints = None
        self.command_waypoints = None

        self.reach_goal_buf = torch.zeros(self.num_envs, device=self.device)
        self.reach_goal_count = torch.zeros(self.num_envs, device=self.device)
        #log
        self.logger = Logger(dt=self.dt)
        self.reward_saved_dict = defaultdict(list)
        self.rewbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.num_reset = 0
        self.num_reach = 0
        
        # load actuator network
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)
        
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_mean_height[:] = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        self.base_quat[:] = self.root_states[:, 3:7]
        base_euler = torch.stack(list(get_euler_xyz(self.base_quat)), dim=1)
        self.base_euler[:] = torch.where(base_euler>math.pi, (-2*math.pi)+base_euler, base_euler)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        self.store_log()            # store log for analysis training progress

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        
        if self.viewer and self.enable_viewer_sync:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            if self.debug_viz:
                self._draw_debug_vis()
            if self.commands_viz:
                self._draw_commands_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)
        
        # remove guide reward term when
        #   1. activate guide_reward_stop flag (in config file)
        #   2. not remove yet
        #   3. each iteration update (step / num_step_per_env)
        #   4. not iteration 0
                # then check
                # 1. if condition_guide_stop == 'task_progress' and tracking position reward term reach xx% (xx = guide_stop_reach)
                # or
                # 2. if condition_guide_stop == 'first_iteration' then remove at iteration 1

        if (self.cfg.rewards.guide_reward_stop and
            "guide" in self.reward_names and 
            (self.common_step_counter % self.cfg.env.num_steps_per_env == 0) and
            (self.common_step_counter // self.cfg.env.num_steps_per_env > 0)):
        
            if  (( self.cfg.rewards.condition_guide_stop == 'task_progress' and
                torch.mean(self.episode_sums["tracking_position"][env_ids]) / self.max_episode_length > self.cfg.rewards.guide_stop_reach * self.reward_scales["tracking_position"]) 
                or
                (self.cfg.rewards.condition_guide_stop == 'first_iteration')):
            
                self.reward_names.remove("guide")
                self.reward_functions.remove(self._reward_guide)
                self.episode_sums["guide"] = torch.zeros(self.num_envs, device=self.device)
                del self.reward_scales["guide"]

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        
        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_radius"] = self.command_ranges["radius"][1]
            self.extras["episode"]["min_command_height"] = self.command_ranges["base_height"][0]
            self.extras["episode"]["reach_rate"] = torch.count_nonzero(self.reach_goal_count[env_ids]) / len(env_ids)
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # reset reach goal count when terminate
        self.reach_goal_count[env_ids] = 0 

        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.
        self.sea_cell_state_per_env[:, env_ids] = 0.

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize buffer for obs and compute reward
        self.commands_scale = torch.tensor([self.obs_scales.height_measurements, 
                                            self.obs_scales.height_measurements, 
                                            self.obs_scales.height_measurements], device=self.device, requires_grad=False,) 
        
        base_euler = torch.stack(list(get_euler_xyz(self.base_quat)), dim=1)
        self.base_euler = torch.where(base_euler>math.pi, (-2*math.pi)+base_euler, base_euler)
        self.base_mean_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(self.num_envs*self.num_actions, 1, 2, device=self.device, requires_grad=False)
        self.sea_hidden_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_cell_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name] 
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        # prepare base observation space
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,  # linear velocity of base express on base frame
                                    self.base_ang_vel  * self.obs_scales.ang_vel, # angular velocity of base express on base frame
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    ), dim=-1)
        
        # add base position => relative position express on base frame or absolute position on env frame
        if self.cfg.env.goal_pos_type == 'relative':
            goal_pos = transformation_inverse(self.base_quat, self.root_states[:, : 3], self.commands) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, goal_pos * self.obs_scales.height_measurements), dim=-1)
        elif self.cfg.env.goal_pos_type == 'absolute':
            goal_pos = self.commands * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, goal_pos), dim=-1)                   # add goal position (express on env frame)
            self.obs_buf = torch.cat((self.obs_buf, self.root_states[:, : 3] * self.obs_scales.height_measurements), dim=-1)   # add base position (express on env frame)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
    def _compute_torques(self, actions):
        # Choose between pd controller and actuator network
        if self.cfg.control.use_actuator_network:
            with torch.inference_mode():
                self.sea_input[:, 0, 0] = (actions * self.cfg.control.action_scale + self.default_dof_pos - self.dof_pos).flatten()
                self.sea_input[:, 0, 1] = self.dof_vel.flatten()
                torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
            return torques
        else:
            # pd controller
            return super()._compute_torques(actions)
        
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        resample_buf = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0 )

        # check reach goal when the agent is in acceptable boundary of the goal state in 3D space
        self.reach_goal_buf = torch.where(euclidean_distance(self.root_states[:, : 3], self.commands) <= self.cfg.commands.accept_error, True, False)
        reach_goal_env_ids = self.reach_goal_buf.nonzero(as_tuple=False).flatten()
        self.reach_goal_count[reach_goal_env_ids] += 1

        if self.cfg.commands.reachgoal_resample: # if flag resample when the agent reach goal activate
            resample_buf |= self.reach_goal_buf  # resample command in env which reach goal
        
        else: # not resample when reach goal => consider can reach goal for only one times.
            self.reach_goal_count = torch.clip(self.reach_goal_count, 0, 1)
        
        # env_ids is id of environment which run equal to resampling_time
        env_ids = resample_buf.nonzero(as_tuple=False).flatten()

        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights() #get height of terrain arround the robot
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
    
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments while run same iteration 
            the command is 3D goal position r 

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """

        # random variable radius(m) and angle (rad)
        r_min = self.command_ranges['radius'][0]
        r_max = self.command_ranges['radius'][1]
        angle_min = self.command_ranges['angle'][0]
        angle_max = self.command_ranges['angle'][1]
        r = torch_rand_float(r_min, r_max, (len(env_ids), 1), device=self.device).squeeze(1)
        angle = torch_rand_float(angle_min, angle_max,  (len(env_ids), 1), device=self.device).squeeze(1)

        # get robot state and offset
        robot_pos_x = self.root_states[env_ids, 0] # x position of robot relative to env
        robot_pos_y = self.root_states[env_ids, 1] # y position of robot relative to env

        # calculate the goal position
        x = robot_pos_x + (torch.cos(angle) * r)
        y = robot_pos_y + (torch.sin(angle) * r)
        z = torch_rand_float(self.command_ranges["base_height"][0], self.command_ranges["base_height"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set new goal point
        self.commands[env_ids, 0] = x
        self.commands[env_ids, 1] = y
        self.commands[env_ids, 2] = z

        self.waypoints_set(env_ids)     # over write command if predefine waypoints
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands range for random
        Args:
            env_ids (List[int]): ids of environments being reset
        """
        
        reach_rate = torch.count_nonzero(self.reach_goal_count[env_ids]) / len(env_ids)
        if  reach_rate > 0.8:
            self.command_ranges["radius"][1] = np.clip(self.command_ranges["radius"][1] + self.cfg.commands.step_radius, 0., self.cfg.commands.max_radius)
            self.command_ranges["base_height"][0] = np.clip(self.command_ranges["base_height"][0] - self.cfg.commands.step_height, self.cfg.commands.min_height, np.inf)
        
        # if torch.mean(self.episode_sums["tracking_height"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_height"]:
        #     self.command_ranges["base_height"][0] = np.clip(self.command_ranges["base_height"][0] - self.cfg.commands.step_height, self.cfg.commands.min_height, np.inf)
        
    def store_log(self): 
        """For store tracking reward, sum reward, ... in logger
        """

        for name in self.reward_names:
            self.reward_saved_dict[name].append(self.extras["episode"]['rew_' + name].item())

        env_ids = self.reset_buf.nonzero(as_tuple=False)    # index of reset environments
        self.cur_reward_sum += self.rew_buf
        self.rewbuffer.extend(self.cur_reward_sum[env_ids][:, 0].cpu().numpy().tolist())
        self.cur_reward_sum[env_ids] = 0
        
        self.num_reset += len(env_ids)
        self.num_reach += torch.count_nonzero(self.reach_goal_count[env_ids])
      
        if self.cfg.env.save_log_steps or self.common_step_counter % self.cfg.env.num_steps_per_env == 0:
            for name in self.reward_saved_dict.keys():
                self.logger.log_states({name : statistics.mean(self.reward_saved_dict[name])})
            self.reward_saved_dict.clear()
            
            if len(self.rewbuffer) > 0: # if rewbuffer is not empty
                self.logger.log_states({'sum_reward' : statistics.mean(self.rewbuffer)})
            else: 
                self.logger.log_states({'sum_reward' : 0})
                
            if "guide" in self.reward_names:
                self.logger.log_states({'guide_reward_active' : 1})
            else:
                self.logger.log_states({'guide_reward_active' : 0})
            
            if self.cfg.terrain.curriculum:
                self.logger.log_states({'terrain_level': self.terrain_levels.float().mean(0)})
                
            if self.cfg.commands.curriculum:
                self.logger.log_states({'radius_level': self.extras["episode"]["max_command_radius"]})
                self.logger.log_states({'height_level': self.extras["episode"]["min_command_height"]})
                
            reach_goal_count = self.num_reach / self.num_reset
            self.logger.log_states({'reach_goal_percent' : reach_goal_count})
            self.num_reset = 0
            self.num_reach = 0
            
            self.logger.log_states({'step_count' : self.common_step_counter})
            self.logger.log_states({'iteration' : self.common_step_counter // self.cfg.env.num_steps_per_env})
    
    def waypoints_set(self, env_ids):
        if self.num_waypoints != None:
            # set command at env_ids
            self.commands[env_ids, 0] = self.command_waypoints[self.idx_waypoints, 0]
            self.commands[env_ids, 1] = self.command_waypoints[self.idx_waypoints, 1]
            self.commands[env_ids, 2] = self.command_waypoints[self.idx_waypoints, 2]

            # update idx command of env_ids (clip in range 0 to num_waypoints)
            self.idx_waypoints[env_ids] += 1
            self.idx_waypoints = torch.clip(self.idx_waypoints, 0, self.num_waypoints - 1)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        if not self.terrain.cfg.measure_heights:
            return
        
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
    
    def _draw_commands_vis(self):
        goal_geom = gymutil.WireframeSphereGeometry(self.cfg.commands.accept_error, 10, 10, None, color=(0, 1, 0))
        for i in range(self.num_envs):
            x = self.commands[i, 0]
            y = self.commands[i, 1]
            z = self.commands[i, 2]
            pose = gymapi.Transform(p = gymapi.Vec3(x, y, z), r = None)
            gymutil.draw_lines(goal_geom, self.gym, self.viewer, self.envs[i], pose)
    
    #------------my additional reward functions----------------
    def _reward_tracking_height(self):
        # Task Reward (tracking base height)
        height_error = euclidean_distance((self.commands[:, 2]).unsqueeze(1), (self.root_states[:, 2]).unsqueeze(1))
        return torch.exp(-(height_error**2)/self.cfg.rewards.tracking_height)
    
    def _reward_tracking_position(self):
        # Task Reward (tracking x-y position)
        distance_error = euclidean_distance(self.commands[:, 0 : 2], self.root_states[:, 0 : 2]) 
        return torch.exp(-(distance_error**2)/self.cfg.rewards.tracking_goal_point)  

    def _reward_stall(self):
        # Task Penalty
        base_lin_vel_xy = self.root_states[:, 7 : 9]
        diff_distance = self.commands[:, 0 : 3] - self.root_states[:, 0 : 3]
        norm_vel = torch.norm(base_lin_vel_xy, dim=1) # norm of vector velocity in only x y axes
        norm_distance = torch.norm(diff_distance, dim=1)

        env_vel_less = torch.where(norm_vel < self.cfg.rewards.velocity_target, 1, 0)
        env_dis_far = torch.where(norm_distance > self.cfg.commands.accept_error, 1, 0)

        return torch.logical_and(env_vel_less, env_dis_far)

    def _reward_guide(self):
        # Reward for guiding the agent move to the right direction
        base_vel_xy = self.root_states[:, 7 : 9]
        diff_distance = self.commands[:, 0 : 2] - self.root_states[:, 0 : 2]
        norm_vel = torch.norm(base_vel_xy, dim=1) # norm of vector velocity in only x y axes
        norm_distance = torch.norm(diff_distance, dim=1)
        
        dot_product = torch.sum(base_vel_xy * diff_distance, dim=1)

        return torch.clip(dot_product / (norm_vel * norm_distance), 0, 1)
    
    def _reward_reach_goal(self):
        return self.reach_goal_buf