import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.export import Dim

def get_reward(state: Tensor)->Tensor:
    agentPos0 = state[:,0:2]
    agentPos1 = state[:,8:10]
    agentDist0 = torch.sqrt(torch.sum(agentPos0**2,dim=1))
    agentDist1 = torch.sqrt(torch.sum(agentPos1**2,dim=1))
    agentDist0 = torch.maximum(agentDist0, torch.tensor(5))
    bladePos0 = state[:,4:6]
    bladePos1 = state[:,12:14]
    bladeDist0 = torch.sqrt(torch.sum(bladePos0**2,dim=1))
    bladeDist0 = torch.minimum(bladeDist0, torch.tensor(5))
    saftey0 = torch.sqrt(torch.sum((bladePos1-agentPos0)**2,dim=1))
    saftey0 = torch.minimum(saftey0, torch.tensor(2))
    reward = agentDist1 - agentDist0 - bladeDist0 + 5*saftey0
    return reward.unsqueeze(1)