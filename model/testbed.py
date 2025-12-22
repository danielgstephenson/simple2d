from world import World, Agent
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_printoptions(sci_mode=False)

class Testbed(World):
    def __init__(self):
        super().__init__(3, device)
        agent0 = Agent(self)
        agent1 = Agent(self)
        agent0.velocity[:] = torch.tensor([+1, 0])
        agent1.velocity[:] = torch.tensor([-1, 0])
        agent0.position[:] = torch.tensor([-3,0])
        agent1.position[:] = torch.tensor([+3,0])
        agent0.blade.position[:] = torch.tensor([-2,-4])
        agent1.blade.position[:] = torch.tensor([+2,+4])
        agent0.blade.velocity[:] = torch.tensor([+5,0])
        agent1.blade.velocity[:] = torch.tensor([-5,0])

testbed = Testbed()

for step in range(50):
    print(step)
    for agent in testbed.agents:
        print(agent.position[0,:].cpu().numpy())
    testbed.step()