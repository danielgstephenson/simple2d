import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.export import Dim
import contextlib
import io

discount = 0.8
self_noise = 0.1
other_noise = 0.01

def get_reward(state: Tensor)->Tensor:
    fighterPos0 = state[:,0:2]
    fighterPos1 = state[:,8:10]
    weaponPos0 = state[:,4:6]
    weaponPos1 = state[:,12:14]
    dist0 = torch.sqrt(torch.sum(fighterPos0**2,dim=1))
    dist1 = torch.sqrt(torch.sum(fighterPos1**2,dim=1))
    close = torch.tensor(5)
    # dist0 = torch.maximum(dist0,close)
    danger0 = torch.sqrt(torch.sum((fighterPos0-weaponPos1)**2,dim=1))
    danger1 = torch.sqrt(torch.sum((fighterPos1-weaponPos0)**2,dim=1))
    reward = dist1 - dist0 # + 0.2 * (danger1 - danger0)
    return reward.unsqueeze(1)

class ValueModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = torch.nn.LeakyReLU(negative_slope=0.01)
        inputSize = 16
        k = 100
        self.hiddenCount = 10
        self.hiddenLayers = nn.ModuleList([nn.Linear(inputSize + i*k, k) for i in range(self.hiddenCount)])
        self.outputLayer = nn.Linear(inputSize + self.hiddenCount*k, 1)
    def forward(self, state: Tensor) -> Tensor:
        x = state
        for i in range(self.hiddenCount):
            h = self.hiddenLayers[i]
            y: Tensor = self.activation(h(x))
            x = torch.cat((x,y),dim=1)
        return self.outputLayer(x)
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)
    
class ActionModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = torch.nn.LeakyReLU(negative_slope=0.01)
        inputSize = 16
        k = 100
        self.hiddenCount = 10
        self.hiddenLayers = nn.ModuleList([nn.Linear(inputSize + i*k, k) for i in range(self.hiddenCount)])
        self.outputLayer = nn.Linear(inputSize + self.hiddenCount*k, 9)
    def forward(self, state: Tensor) -> Tensor:
        x = state
        for i in range(self.hiddenCount):
            h = self.hiddenLayers[i]
            y: Tensor = self.activation(h(x))
            x = torch.cat((x,y),dim=1)
        return self.outputLayer(x)
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str):
    checkpoint = { 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    try:
        torch.save(checkpoint, path)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt detected. Saving checkpoint...')
        torch.save(checkpoint, path)
        print('Checkpoint saved.')
        raise

def save_onnx(model: nn.Module, path: str, device: torch.device):
    with contextlib.redirect_stdout(io.StringIO()):
        example_input = torch.tensor([[i for i in range(16)]],dtype=torch.float32).to(device)
        example_input_tuple = (example_input,)
        onnx_program = torch.onnx.export(
            model, 
            example_input_tuple, 
            dynamo=True,
            input_names=["state"],
            output_names=["output"],
            dynamic_shapes=[[Dim("batch_size"), Dim.AUTO]]
        )
        if onnx_program is not None:
            onnx_program.save(path)