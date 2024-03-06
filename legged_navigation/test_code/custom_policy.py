import sys
sys.path.append("/home/natcha/github/legged_navigation")

import rsl_rl.modules.actor_critic
import torch
import torch.nn as nn
from torch.nn.modules import rnn
from torchsummary import summary
import torchvision
from torchview import draw_graph

num_actions = 12
mlp_input_dim_a = 55
actor_hidden_dims = [512, 256, 128]
activation = nn.ELU()

actor_layers = []
actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
actor_layers.append(activation)
for l in range(len(actor_hidden_dims)):
    if l == len(actor_hidden_dims) - 1:
        actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
    else:
        actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
        actor_layers.append(activation)
actor = nn.Sequential(*actor_layers)

print(actor)

model_graph = draw_graph(actor, input_size=(1, 55), expand_nested=True)
model_graph.visual_graph

# class M(nn.Module):
#     def __init__(self):
#         super(M, self).__init__()
#         self.con2D = nn.Conv2d()
        
#         actor_layers = []
#         actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
#         actor_layers.append(activation)
#         for l in range(len(actor_hidden_dims)):
#             if l == len(actor_hidden_dims) - 1:
#                 actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
#             else:
#                 actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
#                 actor_layers.append(activation)
#         self.actor = nn.Sequential(*actor_layers)
        
#     def forward(self, image_obs, obs):
#         x1 = self.cnn(image)
#         x2 = data
        
#         x = torch.cat((x1, x2), dim=1)
#         x = activation(self.fc1(x))
#         x = self.fc2(x)
#         return x
        

# # model = MyModel()