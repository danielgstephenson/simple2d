from generator import Generator
from model import ValueModel, save_onnx
import torch
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_printoptions(sci_mode=False)

generator = Generator(1,device,steps=5,dtype=torch.float32)
model = ValueModel().to(device).eval()
checkpoint_path = f'./checkpoints/value_checkpoint.pt'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

# print('Saving onnx...')
# save_onnx(model, './onnx/value.onnx', device)

print('Generating...')
start0 = torch.tensor([[-3,0,0,0,-3,0,0,0]],dtype=generator.dtype).to(device)
start1 = torch.tensor([[+3,0,0,0,+3,0,0,0]],dtype=generator.dtype).to(device)
start = torch.cat([start0, start1], dim=1)
outcomes = generator.get_outcomes(start)
outcomeMatrix = outcomes.reshape(81,16)
values = model(outcomeMatrix)
valueMatrix = values.reshape(9,9)
actionValues = torch.amin(valueMatrix,dim=1).unsqueeze(1)
print(actionValues)