from generator import Generator
from model import ValueModel, get_reward, save_checkpoint, save_onnx, discount, other_noise, self_noise
import torch
import torch.nn.functional as F
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_printoptions(sci_mode=False)

model = ValueModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
checkpoint_path = f'./checkpoints/value_checkpoint.pt'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

learning_rate = 0.0001
for param_group in optimizer.param_groups:
    param_group['lr'] = learning_rate

advance = False
target_model =  ValueModel().to(device).eval() if advance else get_reward
if isinstance(target_model, ValueModel):
    target_model.load_state_dict(model.state_dict())

save_onnx(model, './onnx/value.onnx', device)

batch_size = 2000
generator = Generator(batch_size, device)

smooth_loss = 0
loss_smoothing = 0.01
target_smoothing = 0.001
print('Training...')
for batch in range(1000000000000):
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
    save_checkpoint(model, optimizer, checkpoint_path)
    print(f'Batch: {batch}, Loss: {loss_value:05.2f}, Smooth: {smooth_loss:05.2f}')
    if isinstance(target_model, ValueModel):
        with torch.no_grad():
            for target_param, param in zip(target_model.parameters(), model.parameters()):
                new_target_param = target_smoothing * param.data + (1.0 - target_smoothing) * target_param.data
                target_param.data.copy_(new_target_param)
