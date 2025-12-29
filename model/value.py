from sympy import re
from generator import Generator
from reward import get_quality
from save import save_checkpoint, save_onnx
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import os
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_printoptions(sci_mode=False)

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
        x = self.init_layer(x)
        for i in range(self.hidden_count):
            h = self.hidden_layers[i]
            x = + F.silu(h(x))
            # x = #  + torch.sin(h(# ))
            # x = torch.sin(h(x))
            # x = x + F.leaky_relu(h(x),negative_slope=0.01)
        x = self.final_layer(x)
        return x
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)

checkpoint_path = './checkpoints/value_checkpoint.pt'
target_checkpoint_path = './checkpoints/value_checkpoint.pt'
onnx_path = './onnx/value.onnx'
model = ValueModel().to(device)
target_model = ValueModel().to(device).eval()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
discount = 0.9
horizon = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    target_model.load_state_dict(checkpoint['target_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    discount = checkpoint['discount']
    horizon = checkpoint['horizon']
if os.path.exists(target_checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    target_model.load_state_dict(checkpoint['model_state_dict'])
# horizon = 1
# discount = 0.9

if isinstance(target_model, ValueModel):
    target_model.load_state_dict(model.state_dict())

lr = 0.0001
for param_group in optimizer.param_groups:
    param_group['lr'] = lr

print('Saving onnx...')
save_onnx(model, onnx_path, device)

batch_size = 5000 # Reduce to 3000 if GPU memory is limited
generator = Generator(batch_size, device, steps=5)

# quit()

self_noise = 0.2 # 0.2 or 0.3 ?
other_noise = 0.01
smooth_loss = 0
loss_smoothing = 0.05
loss_threshold = 0.02 # If this is negative then the horizon never increases
print('Training...')
for batch in range(1000000000000):
    data = generator.generate()
    old_state = data[:,0:16]
    output = model(old_state)
    with torch.no_grad(): 
        current_quality = get_quality(old_state).repeat_interleave(81,dim=0)
        outcome = data[:,16:].reshape(batch_size*81,16)
        outcome_quality = get_quality(outcome)
        reward = outcome_quality - current_quality
        outcome_value = reward if horizon == 0 else target_model(outcome)
        opponent_options = (1-discount)*reward + discount*outcome_value
        option_matrix = opponent_options.reshape(batch_size,9,9)
        rowMeans = torch.mean(option_matrix,2)
        rowMins = torch.amin(option_matrix,2)
        action_values = other_noise*rowMeans + (1-other_noise)*rowMins
        max_action_value = torch.amax(action_values,1).unsqueeze(1)
        average_action_value = torch.mean(action_values,1).unsqueeze(1)
        target = self_noise*average_action_value + (1-self_noise)*max_action_value
    loss = F.mse_loss(output, target, reduction='mean')
    loss_value = loss.detach().cpu().numpy()
    smooth_loss = loss_smoothing*loss_value + (1-loss_smoothing)*smooth_loss
    if batch == 0: smooth_loss = 2 * loss_value
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    save_checkpoint(model, target_model, optimizer, discount, horizon, checkpoint_path)
    if batch > 50 and np.maximum(loss_value, smooth_loss) < loss_threshold:
        horizon += 1
        smooth_loss = 2 * smooth_loss
        with torch.no_grad():
            for target_param, param in zip(target_model.parameters(), model.parameters()):
                target_param.data.copy_(param.data)
    message = ''
    message += f'Batch: {batch}, '
    message += f'Discount: {discount}, '
    message += f'Horizon: {horizon}, '
    message += f'Loss: {loss_value:07.4f}, '
    message += f'Smooth: {smooth_loss:07.4f}, '
    print(message)
