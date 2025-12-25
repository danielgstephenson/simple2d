from numpy import dtype
from generator import Generator
from model import ValueModel, get_reward, save_checkpoint, save_onnx
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_printoptions(sci_mode=False)

model = ValueModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=100, cooldown=100)
checkpoint_path = f'./checkpoints/value_checkpoint.pt'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

lr = 0.0001
for param_group in optimizer.param_groups:
    param_group['lr'] = lr

advance = False
target_model = ValueModel().to(device).eval() if advance else get_reward
if isinstance(target_model, ValueModel):
    target_model.load_state_dict(model.state_dict())

print('Saving onnx...')
save_onnx(model, './onnx/value.onnx', device)

batch_size = 20000
generator = Generator(batch_size, device)

# quit()

discount = 0.9
self_noise = 0.5
other_noise = 0
smooth_loss = 0
loss_smoothing = 0.01
target_smoothing = 0.001
print('Training...')
for batch in range(1000000000000):
    optimizer.zero_grad()
    data = generator.generate()
    state = data[:,0:16]
    output = model(state)
    reward = get_reward(state)
    outcomes = data[:,16:].reshape(-1,16)  
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
    # scheduler.step(smooth_loss)
    # lr = scheduler.get_last_lr()[0]
    save_checkpoint(model, optimizer, checkpoint_path)
    delta = torch.tensor(0.0).to(device)
    if isinstance(target_model, ValueModel):
        with torch.no_grad():
            for target_param, param in zip(target_model.parameters(), model.parameters()):
                delta += torch.sum((param.data-target_param.data)**2)
                new_target_param = target_smoothing * param.data + (1.0 - target_smoothing) * target_param.data
                target_param.data.copy_(new_target_param)
    delta = torch.sqrt(delta)
    print(f'Batch: {batch}, LR: {lr:.8f}, Loss: {loss_value:07.4f}, Smooth: {smooth_loss:07.4f}, Delta: {delta:05.2f}')
