import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.export import Dim

def get_objective(state: Tensor)->Tensor: 
    agentPos0 = state[:,0:2]
    agentDist0 = torch.sqrt(torch.sum(agentPos0**2,dim=1))
    agentDist0Cost = torch.maximum(agentDist0-10, torch.tensor(0))
    #agentVelocity0 = state[:,2:4]
    #agentSpeed0 = torch.sqrt(torch.sum(agentVelocity0**2,dim=1))
    #bladePos0 = state[:,4:6]
    #bladeDist0 = torch.sqrt(torch.sum(bladePos0**2,dim=1))
    #bladeDist0Cost = torch.maximum(bladeDist0-5, torch.tensor(0))
    #agentSpeed0Reward = torch.where(bladeDist0 < 5, agentSpeed0, 0)
    agentPos1 = state[:,8:10]
    agentDist1 = torch.sqrt(torch.sum(agentPos1**2,dim=1))
    reward = agentDist1 - agentDist0Cost
    return reward.unsqueeze(1)

# Reward(s_t,s_{t+1}) = Objective(s_{t+1}) - Objective(s_t)