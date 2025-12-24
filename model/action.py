from generator import Generator
from model import ValueModel, ActionModel, save_checkpoint, save_onnx, discount, other_noise
import torch
import torch.nn.functional as F
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_printoptions(sci_mode=False)

value_model = ValueModel().to(device).eval()
value_checkpoint_path = './checkpoints/value_checkpoint.pt'
if os.path.exists(value_checkpoint_path):
    checkpoint = torch.load(value_checkpoint_path, weights_only=False)
    value_model.load_state_dict(checkpoint['model_state_dict'])

action_model = ActionModel().to(device)
optimizer = torch.optim.AdamW(action_model.parameters(), lr=0.001)
action_checkpoint_path = './checkpoints/action_checkpoint.pt'
if os.path.exists(action_checkpoint_path):
    checkpoint = torch.load(action_checkpoint_path, weights_only=False)
    action_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

save_onnx(action_model, './action/action.onnx', device)

learning_rate = 0.0001
for param_group in optimizer.param_groups:
    param_group['lr'] = learning_rate

batch_size = 2000
generator = Generator(batch_size, device)

smooth_loss = 0
loss_smoothing = 0.01
print('Training...')
for batch in range(1000000000000):
    data = generator.generate()
    state = data[:,0:16]
    output = action_model(state)
    action_probs = F.softmax(output,dim=1)  
    outcomes = data[:,16:].reshape(-1,16)
    with torch.no_grad():        
        future_values = value_model(outcomes)
        value_matrices = future_values.reshape(batch_size,9,9)
        means = torch.mean(value_matrices,dim=2)
        mins = torch.amin(value_matrices,dim=2)
        value = other_noise*means + (1-other_noise)*mins
        average_value = torch.mean(value,dim=1).unsqueeze(1)
        max_value = torch.amax(value,dim=1).unsqueeze(1)
        min_value = torch.amin(value,dim=1).unsqueeze(1)
        range_value = max_value - min_value
        mean_range_value = torch.mean(range_value)
        target = value - average_value
    loss = F.mse_loss(output, target, reduction='mean')
    loss_value = loss.detach().cpu().numpy()
    smooth_loss = loss_value if batch == 0 else loss_smoothing*loss_value + (1-loss_smoothing)*smooth_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(action_model.parameters(), max_norm=1.0)
    optimizer.step()
    save_checkpoint(action_model, optimizer, action_checkpoint_path)
    print(f'Action Batch: {batch}, Loss: {loss_value:05.2f}, Smooth: {smooth_loss:05.2f}, Range: {mean_range_value:05.2f}')