import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.export import Dim
import math

def get_reward(state: Tensor)->Tensor:
    fighterPos0 = state[:,0:2]
    fighterPos1 = state[:,8:10]
    dist0 = torch.sqrt(torch.sum(fighterPos0**2,dim=1))
    dist1 = torch.sqrt(torch.sum(fighterPos1**2,dim=1))
    dist0 = torch.maximum(dist0,torch.tensor(5))
    # dist1 = torch.maximum(dist1,torch.tensor(5))
    # weaponPos0 = state[:,4:6]
    # weaponPos1 = state[:,12:14]
    # danger0 = torch.sqrt(torch.sum((fighterPos0-weaponPos1)**2,dim=1))
    # danger1 = torch.sqrt(torch.sum((fighterPos1-weaponPos0)**2,dim=1))
    reward = dist1 - dist0 # + 0.2 * (danger1 - danger0)
    return reward.unsqueeze(1)
    
class Siren(nn.Module):
    def __init__(self,in_features: int,out_features: int, w0=1.0, is_first=False, linear=False):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
        self.activation = torch.nn.Identity() if linear else torch.sin
        self.init_weights()
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features,
                     1 / self.in_features
                )
            else:
                bound = math.sqrt(6 / self.in_features) / self.w0
                self.linear.weight.uniform_(-bound, bound)
    def forward(self, x):
        return self.activation(self.linear(x))
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)
    
class SineLinear(nn.Module):
    def __init__(self,in_features: int, out_features: int, w0=1.0):
        super().__init__()
        self.in_features = in_features
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        return torch.sin(self.w0*self.linear(x))
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)

class ValueModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_dim = 16
        k = 500
        self.w0 = 0.1
        self.hidden_count = 4
        self.init_layer = nn.Linear(input_dim, k)
        self.hidden_layers = nn.ModuleList()
        for i in range(self.hidden_count):
            self.hidden_layers.append(nn.Linear(k, k))
        self.final_layer = nn.Linear(k, 1)
    def forward(self, x: Tensor) -> Tensor:
        # x = torch.sin(self.w0 * self.init_layer(x))
        x = self.init_layer(x)
        for i in range(self.hidden_count):
            h = self.hidden_layers[i]
            x = x + F.silu(h(x))
        x = self.final_layer(x)
        return x
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)

# k=100, layer_count = 10, w0_first = 1, w0_hidden = 1
# Batch: 4992, LR: 0.00100000, Loss: 01.6439, Smooth: 02.0420, Delta: 00.00

# k=100, layer_count = 10, w0_first = 30, w0_hidden = 1
# Batch: 2616, LR: 0.00100000, Loss: 207.7644, Smooth: 208.2633, Delta: 00.00

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

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, discount: float, path: str):
    checkpoint = { 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'discount': discount
    }
    try:
        torch.save(checkpoint, path)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt detected. Saving checkpoint...')
        torch.save(checkpoint, path)
        print('Checkpoint saved.')
        raise

def save_onnx(model: nn.Module, path: str, device: torch.device):
    # with contextlib.redirect_stdout(io.StringIO()):
    example_input = torch.tensor([[i for i in range(16)]],dtype=torch.float32).to(device)
    example_input_tuple = (example_input,)
    onnx_program = torch.onnx.export(
        model, 
        example_input_tuple, 
        dynamo=True,
        input_names=["state"],
        output_names=["output"],
        dynamic_shapes=[[Dim("batch_size"), Dim.AUTO]],
        verbose=False
    )
    if onnx_program is not None:
        onnx_program.save(path)