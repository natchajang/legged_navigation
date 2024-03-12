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
from PIL import Image
import cv2

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from collections import defaultdict, deque

from legged_navigation import LEGGED_GYM_ROOT_DIR
from legged_navigation.envs import LeggedRobot
from legged_navigation.utils.terrain import Terrain
from .navigation.anymal_c_nav_config import AnymalCNavCfg
from legged_navigation.utils.math import quat_apply_yaw
from legged_navigation.utils import Logger

class AnymalNav(LeggedRobot):
    cfg : AnymalCNavCfg
    
    #------------- Init --------------
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # allocate buffers
        if self.cfg.camera.active:
            if self.cfg.camera.image_type == gymapi.IMAGE_DEPTH: 
                self.camera_obs_buf = torch.zeros(self.num_envs, 1, self.cfg.camera.img_height, self.cfg.camera.img_width, device=self.device, dtype=torch.float)
            elif self.cfg.camera.image_type == gymapi.IMAGE_COLOR:
                self.camera_obs_buf = torch.zeros(self.num_envs, 4, self.cfg.camera.img_height, self.cfg.camera.img_width, device=self.device, dtype=torch.float)

        # addition attributes
        self.commands_viz = False                           # for visualizing command (plane of base height / direction and ampitude of velocity / axis of desired base orientation)
        self.command_set_flag = False                       # flag for setting command without resample
        self.camera_image_viz = True                        # flag for visualizing image from camera sensor

        # for log save
        self.logger = Logger(self.dt)                       # logger for saving training history
        self.reward_saved_dict = defaultdict(list)          
        self.count_steps = 0                                # for count step if mod self.num_steps_per_env if = 0 save log of reward
        self.cur_reward_sum = torch.zeros(self.cfg.env.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.rewbuffer = deque(maxlen=100)
        self.command_level = np.array([self.cfg.commands.ranges.start_vel,
                                       self.cfg.commands.ranges.start_height,
                                       self.cfg.commands.ranges.start_angle*math.pi], dtype=float)   # start command range linear_velocity, base height, orientation
        
        # load actuator network
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)
            
    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize buffer for obs and compute reward
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, 
                                            self.obs_scales.lin_vel, 
                                            self.obs_scales.height_measurements, 
                                            self.obs_scales.dof_pos,
                                            self.obs_scales.dof_pos,
                                            self.obs_scales.dof_pos], device=self.device, requires_grad=False,) 
        
        base_euler = torch.stack(list(get_euler_xyz(self.base_quat)), dim=1)
        self.base_euler = torch.where(base_euler > math.pi, (-2 * math.pi) + base_euler, base_euler) # change range
        self.base_mean_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        
        # first command
        self.command_own = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=float, device=self.device, requires_grad=False)
        
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(self.num_envs*self.num_actions, 1, 2, device=self.device, requires_grad=False)
        self.sea_hidden_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_cell_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)
        
        # camera sensor buffer
        if self.cfg.camera.active:
            self.camera_tensor_list = [] # list to collect each camera tensor

            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            
            for i, c in enumerate(self.camera_handles):
                _camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], c, self.cfg.camera.image_type)
                camera_tensor = gymtorch.wrap_tensor(_camera_tensor)
                self.camera_tensor_list.append(camera_tensor)
                
            self.camera_images = torch.stack(self.camera_tensor_list) # camera data tensor size (num_env, height, width, 1 or 4)
            self.gym.end_access_image_tensors(self.sim)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()
    
    def _create_envs(self):
        """ Creates environments: call in __init_buffer()
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        
        # camera Sensor Preset properties/ attaching Rigid body / local transform
        if self.cfg.camera.active:
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg.camera.img_width
            camera_props.height = self.cfg.camera.img_height
            camera_props.horizontal_fov = self.cfg.camera.horizontal_fov
            camera_props.near_plane = self.cfg.camera.near_plane
            camera_props.far_plane = self.cfg.camera.far_plane
            camera_props.enable_tensors = self.cfg.camera.enable_tensors
            
            rigid_idx = self.gym.find_asset_rigid_body_index(robot_asset, self.cfg.camera.attach_rigid_name)
            
            camera_local_transform = gymapi.Transform()
            camera_local_transform.p = self.cfg.camera.offset_position
            camera_local_transform.r = self.cfg.camera.offset_rotation

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.camera_handles = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            
            if self.cfg.camera.active:
                camera_handle = self.gym.create_camera_sensor(env_handle, camera_props) # create camera handle
                rigid_handle = self.gym.get_actor_rigid_body_handle(env_handle, actor_handle, rigid_idx) # create attach rigid body
                self.gym.attach_camera_to_body(camera_handle, env_handle, rigid_handle, camera_local_transform, gymapi.FOLLOW_TRANSFORM) # attach camera to rigid body
                self.camera_handles.append(camera_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
    
    #------------- Step --------------
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if self.cfg.camera.return_camera_obs:
            obs, privileged_obs, camera_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
            return obs, privileged_obs, camera_obs
        else: 
            obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
            return obs, privileged_obs

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        if self.cfg.camera.return_camera_obs:
            return self.obs_buf, self.privileged_obs_buf, self.camera_obs_buf, self.rew_buf, self.reset_buf, self.extras
        else:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        # update image data from camera tensor list
        if self.cfg.camera.active:
            if not self.viewer:
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)

            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            self.camera_images = torch.stack(self.camera_tensor_list)
            self.gym.end_access_image_tensors(self.sim)

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
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        self.store_log()
        self.count_steps += 1       # count step
        
        if self.viewer and self.enable_viewer_sync:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            if self.debug_viz:
                self._draw_debug_vis()
            if self.commands_viz:
                self._draw_commands_vis()
            if self.camera_image_viz:
                self._camera_image_vis(0)
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.
        self.sea_cell_state_per_env[:, env_ids] = 0.
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  (self.base_mean_height * self.obs_scales.height_measurements).unsqueeze(1),
                                    self.base_euler * self.obs_scales.dof_pos,
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.commands[:, :] * self.commands_scale,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    ), dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        # add camera sensor data
        if self.cfg.camera.active:
            self.image_preprocessing()
            self.camera_obs_buf = self.camera_images

    def get_camera_observation(self):
        return self.camera_obs_buf
    
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
    
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments while run same iteration

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # random command linear velocity (m/s) and base height (m)
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["base_height"][0], self.command_ranges["base_height"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        # random roll pitch yaw command in range -x to x rad
        roll_command = torch_rand_float(self.command_ranges["base_roll"][0], self.command_ranges["base_roll"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        pitch_command = torch_rand_float(self.command_ranges["base_pitch"][0], self.command_ranges["base_pitch"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        yaw_command = torch_rand_float(self.command_ranges["base_yaw"][0], self.command_ranges["base_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        # add orientation to command buffer
        self.commands[env_ids, 3] = roll_command
        self.commands[env_ids, 4] = pitch_command
        self.commands[env_ids, 5] = yaw_command
        
        self.commands_set()
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands range for random
        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands (vel)
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            # change range of linear velocity of x axis
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - self.cfg.commands.step_vel, -self.cfg.commands.max_vel, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + self.cfg.commands.step_vel, 0., self.cfg.commands.max_vel)
            
            self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - self.cfg.commands.step_vel, -self.cfg.commands.max_vel, 0.)
            self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + self.cfg.commands.step_vel, 0., self.cfg.commands.max_vel)

            self.command_level[0] = max(self.command_level[0], self.command_ranges["lin_vel_x"][1])
            
        # If the tracking reward is above 80% of the maximum, increase the range of commands (height)
        if torch.mean(self.episode_sums["tracking_height"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_height"]:
            # change range of linear velocity of x axis
            self.command_ranges["base_height"][0] = np.clip(self.command_ranges["base_height"][0] - self.cfg.commands.step_height, self.cfg.commands.min_height, self.command_ranges["base_height"][1])
            self.command_level[1] = min(self.command_level[1], self.command_ranges["base_height"][0])
            
        if torch.mean(self.episode_sums["tracking_orientation"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_orientation"]:
            # change range of linear velocity of x axis
            min_angle = np.clip(self.command_ranges["base_roll"][0] - self.cfg.commands.step_angle, -self.cfg.commands.max_angle, 0.)
            max_angle = np.clip(self.command_ranges["base_roll"][1] + self.cfg.commands.step_angle, 0., self.cfg.commands.max_angle)
            self.command_ranges["base_roll"][0] = min_angle
            self.command_ranges["base_roll"][1] = max_angle
            self.command_ranges["base_pitch"][0] = min_angle
            self.command_ranges["base_pitch"][1] = max_angle
            self.command_ranges["base_yaw"][0] = min_angle
            self.command_ranges["base_yaw"][1] = max_angle
            
            self.command_level[2] = max(self.command_level[2], max_angle)
        
    def commands_set(self):
        if self.command_set_flag:
            self.commands = self.command_own
        
    def store_log(self): 
        """For store tracking reward and sum reward in logger
        """
        for name in self.reward_names:
            if 'tracking' in name:
                self.reward_saved_dict[name].append(self.extras["episode"]['rew_' + name].item())

        # calulate sum reward for logger
        self.cur_reward_sum += self.rew_buf
        new_ids = (self.reset_buf > 0).nonzero(as_tuple=False)
        self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        self.cur_reward_sum[new_ids] = 0
          
        # add for saving sum reward
        if self.count_steps % self.cfg.env.num_steps_per_env == 0:
            for name in self.reward_saved_dict.keys():
                self.logger.log_states({name : statistics.mean(self.reward_saved_dict[name])})
            self.reward_saved_dict.clear()
            
            if len(self.rewbuffer) != 0: # if rewbuffer is not empty
                self.logger.log_states({'sum_reward' : statistics.mean(self.rewbuffer)})
            else: 
                self.logger.log_states({'sum_reward' : 0})

            self.logger.log_states({'step_count' : self.count_steps})
            self.logger.log_states({'iteration' : self.count_steps / self.cfg.env.num_steps_per_env})
            
            self.logger.log_states({'command_level': self.command_level.copy()})
            if self.cfg.terrain.curriculum == True:
                self.logger.log_states({'terrain_level': self.terrain_levels.float().mean(0)})

    #------------- image preprocess ----------------
    def image_preprocessing(self):
        if self.cfg.camera.image_type == gymapi.IMAGE_DEPTH: 
            # depth image is the negative distance from camera to pixel in view direction in world coordinate units (meters)
            # print("0. Min: {} / Max: {}".format(torch.min(self.camera_images), torch.max(self.camera_images)))
            # 1. -inf implies no depth value, set it to zero. output will be black.
            self.camera_images[self.camera_images == -np.inf] = 0
            # print("1. Min: {} / Max: {}".format(torch.min(self.camera_images), torch.max(self.camera_images)))
            # # 2. clamp depth image to xx(cfg.camera.clamp_distance) meters to make output image human friendly
            self.camera_images[self.camera_images < -self.cfg.camera.clamp_distance] = -self.cfg.camera.clamp_distance
            # print("2. Min: {} / Max: {}".format(torch.min(self.camera_images), torch.max(self.camera_images)))
            # 3. flip the direction so near-objects are light and far objects are dark
            self.camera_images = -255.0*(self.camera_images/torch.min(self.camera_images + 1e-4))
            # print("3. Min: {} / Max: {}".format(torch.min(self.camera_images), torch.max(self.camera_images)))
            # 4. add dimension x (batch_size, x(=1), img_height, img_width)
            self.camera_images = self.camera_images.unsqueeze(1)
            
    #------------- Visualization ----------------    
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
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
        plane_geom_command = gymutil.WireframeBoxGeometry(2, 2, 0.005, None, color=(0, 1, 0))
        plane_geom_measure = gymutil.WireframeBoxGeometry(2, 2, 0.005, None, color=(1, 1, 0))
        axes_geom = gymutil.AxesGeometry(scale=1, pose=None)
        for i in range(self.num_envs):
            # measurement
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            base_quat = (self.root_states[i, 3:7]).cpu().numpy()
            base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])[0].cpu().numpy() # ref to base frame
            x = base_pos[0]
            y = base_pos[1]
            z = base_pos[2]
            rx = base_quat[0]
            ry = base_quat[1]
            rz = base_quat[2]
            rw = base_quat[3]
            vx = base_lin_vel[0]
            vy = base_lin_vel[1]
            vz = base_lin_vel[2]
            
            # commands
            command_height = self.commands[i, 2]
            command_vel = self.commands[i, :2].cpu().numpy()
            command_orien = self.commands[i, 3:]
            command_quat = quat_from_euler_xyz(command_orien[0], command_orien[1], command_orien[2])
      
            # Draw plane height
            plane_pose_measure = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
            gymutil.draw_lines(plane_geom_measure, self.gym, self.viewer, self.envs[i], plane_pose_measure)
            plane_pose_command = gymapi.Transform(gymapi.Vec3(x, y, command_height), r=None)
            gymutil.draw_lines(plane_geom_command, self.gym, self.viewer, self.envs[i], plane_pose_command)
            
            # Draw linear velocity direction
            # measurement
            vel_transform = gymapi.Transform(gymapi.Vec3(x, y, z), r=gymapi.Quat(rx, ry, rz, rw))
            vel_measure = vel_transform.transform_point(gymapi.Vec3(vx, vy, vz))
            line = [x, y, z+0.3, vel_measure.x, vel_measure.y, vel_measure.z+0.3]
            line_color = [1, 0, 1]
            self.gym.add_lines(self.viewer, self.envs[i], 1, line, line_color)
            # command
            vel_transform_command = gymapi.Transform(gymapi.Vec3(x, y, z), r=gymapi.Quat(rx, ry, rz, rw))
            vel_target = vel_transform_command.transform_point(gymapi.Vec3(command_vel[0], command_vel[1], 0))
            line = [x, y, z+0.3, vel_target.x, vel_target.y, vel_target.z+0.3]
            line_color = [0, 0.5, 0.5]
            self.gym.add_lines(self.viewer, self.envs[i], 1, line, line_color)
            
            # Draw orientation
            axes_pose = gymapi.Transform(gymapi.Vec3(x, y, z), gymapi.Quat(command_quat[0], command_quat[1], command_quat[2], command_quat[3]))
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], axes_pose)

    # function to show image from camera in specific environment id
    def _camera_image_vis(self, env_id):
        # locate camera position and orietation (if enable this axes will appear in image as well !!!)
        # cam_position = self.gym.get_camera_transform(self.sim, self.envs[env_id], self.actor_handles[env_id])
        # axes_geom = gymutil.AxesGeometry(scale=0.5, pose=None)
        # axes_pose = gymapi.Transform(cam_position.p, cam_position.r)
        # gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[env_id], axes_pose)

        # show image
        if self.cfg.camera.image_type == gymapi.IMAGE_DEPTH:
            frame = self.camera_images[env_id].flatten(0, 1).cpu().numpy().astype(np.uint8)
        elif self.cfg.camera.image_type == gymapi.IMAGE_COLOR:
            frame = self.camera_images[env_id, :, :, :3].cpu().numpy()
        cv2.imshow('Frame', frame) 
        cv2.waitKey(1) 
    
    #------------ My Additional Reward Functions ----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        if self.cfg.rewards.reward_tracking == 'binary': 
            return  torch.where(lin_vel_error<=self.cfg.rewards.reward_tracking_accept['velocity'], 1., 0.)
        if self.cfg.rewards.reward_tracking == 'progress_estimator': 
            return torch.clip(1-(lin_vel_error/self.cfg.rewards.reward_tracking_accept['velocity']), 0, 1)
        if self.cfg.rewards.reward_tracking == 'gaussian': 
            return torch.exp(-(lin_vel_error**2)/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_height(self):
        # Penalize base height away from target which random for each iteration
        # command order => lin_vel_x, lin_vel_y, base_height, roll, pitch, yaw
        height_error = torch.square(self.commands[:, 2] - self.base_mean_height)
        if self.cfg.rewards.reward_tracking == 'binary': 
            return torch.where(height_error<=self.cfg.rewards.reward_tracking_accept['height'], 1., 0.)
        if self.cfg.rewards.reward_tracking == 'progress_estimator': 
            return torch.clip(1-(height_error/self.cfg.rewards.reward_tracking_accept['height']), 0, 1)
        if self.cfg.rewards.reward_tracking == 'gaussian': 
            return torch.exp(-(height_error**2)/self.cfg.rewards.tracking_height)
    
    def _reward_tracking_orientation(self):
        # Penalize base orientation away from target while walking
        # command order => lin_vel_x, lin_vel_y, base_height, roll, pitch, yaw
        orientation_error = torch.sum(torch.square(self.commands[:, 3:] - self.base_euler), dim=1) # error between command and measurement
        if self.cfg.rewards.reward_tracking == 'binary': 
            return torch.where(orientation_error<=self.cfg.rewards.reward_tracking_accept['orientation'], 1., 0.)
        if self.cfg.rewards.reward_tracking == 'progress_estimator': 
            return torch.clip(1-(orientation_error/self.cfg.rewards.reward_tracking_accept['orientation']), 0, 1)
        if self.cfg.rewards.reward_tracking == 'gaussian': 
            return torch.exp(-(orientation_error**2)/(self.cfg.rewards.tracking_orientation))
        
    