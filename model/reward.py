import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.export import Dim

def get_quality(state: Tensor)->Tensor: 
    agentPos0 = state[:,0:2]
    agentDist0 = torch.sqrt(torch.sum(agentPos0**2,dim=1))
    agentFar0 = torch.maximum(agentDist0-10, torch.tensor(0))
    agentPos1 = state[:,8:10]
    agentDist1 = torch.sqrt(torch.sum(agentPos1**2,dim=1))
    reward = agentDist1 - agentFar0
    return reward.unsqueeze(1)

def get_reward(old_state: Tensor, new_state: Tensor)->Tensor:
    old_quality = get_quality(old_state)
    new_quality = get_quality(new_state)
    return new_quality - old_quality