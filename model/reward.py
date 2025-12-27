import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.export import Dim

def get_reward(state: Tensor)->Tensor:
    fighterPos0 = state[:,0:2]
    fighterPos1 = state[:,8:10]
    dist0 = torch.sqrt(torch.sum(fighterPos0**2,dim=1))
    dist1 = torch.sqrt(torch.sum(fighterPos1**2,dim=1))
    dist0 = torch.maximum(dist0,torch.tensor(10))
    # dist1 = torch.maximum(dist1,torch.tensor(5))
    # weaponPos0 = state[:,4:6]
    weaponPos1 = state[:,12:14]
    danger0 = torch.sqrt(torch.sum((fighterPos0-weaponPos1)**2,dim=1))
    danger0 = torch.minimum(dist0,torch.tensor(2))
    # danger1 = torch.sqrt(torch.sum((fighterPos1-weaponPos0)**2,dim=1))
    reward = dist1 - dist0 - danger0
    return reward.unsqueeze(1)