from generator import Generator
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import contextlib
import io
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_printoptions(sci_mode=False)

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
    
def get_reward(state: Tensor)->Tensor:
    fighterPos0 = state[:,0:2]
    fighterPos1 = state[:,8:10]
    weaponPos0 = state[:,4:6]
    weaponPos1 = state[:,12:14]
    dist0 = torch.sqrt(torch.sum(fighterPos0**2,dim=1))
    dist1 = torch.sqrt(torch.sum(fighterPos1**2,dim=1))
    close = torch.tensor(5)
    dist0 = torch.maximum(dist0,close)
    danger0 = torch.sqrt(torch.sum((fighterPos0-weaponPos1)**2,dim=1))
    danger1 = torch.sqrt(torch.sum((fighterPos1-weaponPos0)**2,dim=1))
    reward = dist1 - dist0 + 0.2 * (danger1 - danger0)
    return reward.unsqueeze(1)

def save_onnx(model: nn.Module, path: str):
    with contextlib.redirect_stdout(io.StringIO()):
        example_input = torch.tensor([[i for i in range(16)]],dtype=torch.float32).to(device)
        example_input_tuple = (example_input,)
        onnx_program = torch.onnx.export(model, example_input_tuple, dynamo=True)
        if onnx_program is not None:
            onnx_program.save(path)

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


model = ValueModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
checkpoint_path = f'./checkpoints/checkpoint.pt'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

learning_rate = 0.0001
for param_group in optimizer.param_groups:
    param_group['lr'] = learning_rate

advance = True
old_model = get_reward
if advance:
    old_model = ValueModel().to(device)

batch_size = 2000
generator = Generator(batch_size, device)

smooth_loss = 0
smoothing = 0.01
discount = 0.95
self_noise = 0.1
other_noise = 0.01
print('Training...')
for epoch in range(100000000000):
    save_checkpoint(model, optimizer, checkpoint_path)
    if advance:
        if isinstance(old_model, ValueModel):
            old_model.load_state_dict(model.state_dict())
    for batch in range(1000):
        data = generator.generate()
        state = data[:,0:16]
        output = model(state)
        reward = get_reward(state)
        outcomes = data[:,16:].reshape(-1,16)  
        with torch.no_grad():        
            future_values = old_model(outcomes)
            value_matrices = future_values.reshape(batch_size,9,9)
            means = torch.mean(value_matrices,2)
            mins = torch.amin(value_matrices,2)
            action_values = other_noise*means + (1-other_noise)*mins
            max_value = torch.amax(action_values,1).unsqueeze(1)
            average_value = torch.mean(action_values,1).unsqueeze(1)
            future_value = self_noise*average_value + (1-self_noise)*max_value
            target = (1-discount)*reward + discount*future_value
        loss = F.mse_loss(output, target, reduction='mean')
        loss_value = loss.detach().cpu().numpy()
        smooth_loss = loss_value if batch == 0 else smoothing*loss_value + (1-smoothing)*smooth_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print(f'Epoch: {epoch}, Batch: {batch}, Loss: {loss_value:05.2f}, Smooth: {smooth_loss:05.2f}')
