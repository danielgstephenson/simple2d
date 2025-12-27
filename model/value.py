from generator import Generator
from reward import get_reward
from save import save_checkpoint, save_onnx
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import os

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
            x = x + F.silu(h(x))
        x = self.final_layer(x)
        return x
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)


checkpoint_path = './checkpoints/value_checkpoint.pt'
onnx_path = './onnx/value.onnx'
model = ValueModel().to(device)
target_model = ValueModel().to(device).eval()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
discount = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    discount = checkpoint['discount']

if isinstance(target_model, ValueModel):
    target_model.load_state_dict(model.state_dict())

lr = 0.00001
for param_group in optimizer.param_groups:
    param_group['lr'] = lr

print('Saving onnx...')
save_onnx(model, onnx_path, device)

batch_size = 3000 # Use 3000 on the local machine because of GPU memory limits
generator = Generator(batch_size, device, steps=5)

# quit()

self_noise = 0.1
other_noise = 0.1
smooth_loss = 0
loss_smoothing = 0.01
print('Training...')
for batch in range(1000000000000):
    data = generator.generate()
    state = data[:,0:16]
    output = model(state)
    reward = get_reward(state)
    outcomes = data[:,16:].reshape(-1,16)
    if discount == 0:
        target = reward
    else:
        with torch.no_grad():        
            future_values = target_model(outcomes)
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
    smooth_loss = loss_value if batch == 0 else loss_smoothing*loss_value + (1-loss_smoothing)*smooth_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    save_checkpoint(model, optimizer, discount, checkpoint_path)
    delta = torch.tensor(0.0).to(device)
    if isinstance(target_model, ValueModel) and batch > 100:
        if loss_value < 0.06:
            discount = min(0.99, discount + 0.0001)
            with torch.no_grad():
                for target_param, param in zip(target_model.parameters(), model.parameters()):
                    new_target_param = param.data
                    delta += torch.sum((new_target_param-target_param.data)**2)
                    target_param.data.copy_(new_target_param)
        else:
            discount = max(0.0, discount - 0.0001)
    print(f'Batch: {batch}, Discount: {discount:.4f}, Loss: {loss_value:07.4f}, Smooth: {smooth_loss:07.4f}')
