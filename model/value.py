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
checkpoint_path = f'./checkpoints/value_checkpoint0.pt'
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

# print('Saving onnx...')
# save_onnx(model, './onnx/value.onnx', device)

batch_size = 5
print('start:')
generator = Generator(batch_size, device)

start = generator.get_start()
testData = torch.tensor([[
    -1.5383503606070938, 1.6556488490003125,
    -1.2060559371513624, 0.5890940403668103,
    -1.4174889692338488, 2.4769011919216943,
    -3.1537424456779806, 0.9168236340224539,
    -2.690579221153769, 0,
    1.349066778281722, 0,
    -2.9942584965125607, 0,
    0.04695504633279549, 0
]],dtype=torch.float32).to(device)
# print('start.shape', start.shape)
# print('testData.shape', testData.shape)
# testData = torch.cat([testData, start], dim=0)
# reward = get_reward(testData)
# output = model(testData)
# sqError = (reward - output) ** 2
# results = torch.cat([reward, output, sqError], dim=1)
# print('positions:')
# print(testData[:,0:2])
# print('results:')
# print(results)

quit()

discount = 0.1
self_noise = 1.0
other_noise = 1.0
smooth_loss = 0
loss_smoothing = 0.05
target_smoothing = 0.001
print('Training...')
for batch in range(1000000000000):
    optimizer.zero_grad()
    start = generator.generate()
    # data = testData
    state = start[:,0:16]
    output = model(state)
    reward = get_reward(state)
    # outcomes = data[:,16:].reshape(-1,16)  
    # with torch.no_grad():        
    #     future_values = target_model(outcomes)
    #     value_matrices = future_values.reshape(batch_size,9,9)
    #     means = torch.mean(value_matrices,2)
    #     mins = torch.amin(value_matrices,2)
    #     action_values = other_noise*means + (1-other_noise)*mins
    #     max_value = torch.amax(action_values,1).unsqueeze(1)
    #     average_value = torch.mean(action_values,1).unsqueeze(1)
    #     future_value = self_noise*average_value + (1-self_noise)*max_value
    #     target = (1-discount)*reward + discount*future_value
    target = reward
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
