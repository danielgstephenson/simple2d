from simulation import World, Agent
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
defaultType = torch.float64
torch.set_default_dtype(defaultType)
torch.set_printoptions(sci_mode=False)

class Testbed(World):
    def __init__(self):
        super().__init__(2)
        agent1 = Agent(self)
        agent2 = Agent(self)
        agent1.velocity[:] = torch.tensor([+1, 0])
        agent2.velocity[:] = torch.tensor([-1, 0])
        agent1.position[:] = torch.tensor([-3,0])
        agent2.position[:] = torch.tensor([+3,0])
        agent1.blade.position[:] = torch.tensor([-2,-4])
        agent2.blade.position[:] = torch.tensor([+2,+4])
        agent1.blade.velocity[:] = torch.tensor([+5,0])
        agent2.blade.velocity[:] = torch.tensor([-5,0])

testbed = Testbed()

for step in range(50):
    print(step)
    # for blade in testbed.blades:
    #     print(blade.position[0,:].cpu().numpy())
    for agent in testbed.agents:
        print(agent.position[0,:].cpu().numpy())
    testbed.step()