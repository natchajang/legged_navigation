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

import numpy as np

from isaacgym import gymapi

import torch
import torch.nn as nn
from torch.distributions import Normal
from .actor_critic import get_activation

class ActorCriticCNN(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs): # kwargs={'camera_cfg':AnymalCNavCfg.camera ,'policy_cfg':AnymalCNavCfgPPO.policy}
        
        super(ActorCriticCNN, self).__init__()

        if kwargs and kwargs['camera_cfg'].active == True:
            print("ActorCritic CNN Structure: " + str([key for key in kwargs.keys()]))
            self.actor = CNNNet(img_type = kwargs['camera_cfg'].image_type,
                                img_height = kwargs['camera_cfg'].img_height,
                                img_width = kwargs['camera_cfg'].img_width,
                                num_state_obs = num_actor_obs,
                                num_actions = num_actions,
                                con2D_parameters = kwargs['policy_cfg']["actor_con2D_parameters"],
                                combine_hidden_dims = actor_hidden_dims,
                                activation = activation)
            
            self.critic = CNNNet(img_type = kwargs['camera_cfg'].image_type,
                                img_height = kwargs['camera_cfg'].img_height,
                                img_width = kwargs['camera_cfg'].img_width,
                                num_state_obs = num_critic_obs,
                                num_actions = 1,
                                con2D_parameters = kwargs['policy_cfg']["critic_con2D_parameters"],
                                combine_hidden_dims = critic_hidden_dims,
                                activation = activation)

        else:
            raise Exception("camera config active flag in kwargs is False") 

        print(f"Actor Structure: {self.actor}")
        print(f"Critic Structure: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, camera_observations):
        mean = self.actor(observations, camera_observations)       
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, camera_observations):
        self.update_distribution(observations, camera_observations)      
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, camera_observations):
        actions_mean = self.actor(observations, camera_observations)      
        return actions_mean

    def evaluate(self, critic_observations, camera_observations):
        value = self.critic(critic_observations, camera_observations) 
        return value

class CNNNet(nn.Module):
    def __init__(self,  img_type,
                        img_height,
                        img_width,
                        num_state_obs,
                        num_actions = 12,
                        con2D_parameters = [[10, (5,5), (5,5)]], #  [[channels_out_1, kernel_size_1/ stride_1], ... [...]]
                        combine_hidden_dims = [512, 256, 128],
                        activation = 'elu'):
        
        super(CNNNet, self).__init__()

        #------------------- Image branch -------------------
        if img_type == gymapi.IMAGE_DEPTH:
            in_channels = 1
        elif img_type == gymapi.IMAGE_COLOR:
            in_channels = 3

        conv2D_layers = []
        out_channels = con2D_parameters[0][0]
        kernel_size = con2D_parameters[0][1]
        stride_size = con2D_parameters[0][2]
        h = np.floor(((img_height - kernel_size[0]) / stride_size[0]) + 1)
        w = np.floor(((img_width - kernel_size[1]) / stride_size[1]) + 1)
        
        conv2D_layers.append(nn.Conv2d(in_channels=in_channels, 
                                       out_channels=out_channels, 
                                       kernel_size=kernel_size, 
                                       stride=stride_size,
                                       padding=(0, 0),
                                       dilation=(1,1)))
        conv2D_layers.append(get_activation(activation))
        conv2D_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        h = np.floor(((h - 3) / 2) + 1)
        w = np.floor(((w - 3) / 2) + 1)
        conv2D_layers.append(nn.Dropout())

        for l in range(1, len(con2D_parameters)):
            in_channels = con2D_parameters[l-1][0]
            out_channels = con2D_parameters[l][0]
            kernel_size = con2D_parameters[l][1]
            stride_size = con2D_parameters[l][2]
            conv2D_layers.append(nn.Conv2d(in_channels=in_channels, 
                                            out_channels=out_channels, 
                                            kernel_size=kernel_size, 
                                            stride=stride_size,
                                            padding=(0, 0),
                                            dilation=(1,1)))
            h = np.floor(((h - kernel_size[0]) / stride_size[0]) + 1)
            w = np.floor(((w - kernel_size[1]) / stride_size[1]) + 1)
            conv2D_layers.append(get_activation(activation))
            conv2D_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            h = np.floor(((h - 3) / 2) + 1)
            w = np.floor(((w - 3) / 2) + 1)
            conv2D_layers.append(nn.Dropout())

        conv2D_layers.append(nn.Flatten())
        conv2D_layers.append(get_activation(activation))
        self.conv2D = nn.Sequential(*conv2D_layers)

        #------------------- Layers after combine -------------------
        mlp_input_dim = int(h * w * con2D_parameters[-1][0])

        fc_layers = []
        fc_layers.append(nn.Linear(mlp_input_dim + num_state_obs, combine_hidden_dims[0]))
        fc_layers.append(get_activation(activation))
        for l in range(len(combine_hidden_dims)):
            if l == len(combine_hidden_dims) - 1:
                fc_layers.append(nn.Linear(combine_hidden_dims[l], num_actions))
            else:
                fc_layers.append(nn.Linear(combine_hidden_dims[l], combine_hidden_dims[l + 1]))
                fc_layers.append(get_activation(activation))

        self.mlp = nn.Sequential(*fc_layers)

    def forward(self, state, img):
        # Image branch
        img = self.conv2D(img) # 576
     
        # Combine image and state
        x = torch.cat([img, state], dim=1)
        x = self.mlp(x)

        return x