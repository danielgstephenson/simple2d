from world import World, Agent
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_printoptions(sci_mode=False)

class Generator(World):
    def __init__(self, sample_size: int, device: torch.device, steps = 5, dtype = torch.float32):
        super().__init__(sample_size*81, device, dtype)
        self.sample_size = sample_size
        self.steps = steps
        agent0 = Agent(self)
        agent1 = Agent(self)
        actionPairs = torch.cartesian_prod(self.actions,self.actions)
        actionMatrix = actionPairs.repeat(self.sample_size,1)
        agent0.action = actionMatrix[:,0]
        agent1.action = actionMatrix[:,1]
        self.maxSpawnDistances = torch.tensor([5,10,15,20,30,50],dtype=self.dtype).to(device)
        self.maxAgentSpeeds = torch.tensor([1,2,4,7],dtype=self.dtype).to(device)
        self.maxReaches = torch.tensor([2,5,10],dtype=self.dtype).to(device)
        self.maxBladeSpeeds = torch.tensor([2,5,10,20],dtype=self.dtype).to(device)

    def rdir(self, n: int):
        vectors = torch.randn(n, 2, device=self.device)
        dirs = F.normalize(vectors, p=2, dim=1)
        return dirs
    
    def runif(self, n: int):
        return torch.rand(n, device=self.device).unsqueeze(1)
    
    def choose(self, source, n):
        return torch.multinomial(
            input=source, 
            num_samples=n, 
            replacement=True).unsqueeze(1)
    
    def getAgentStart(self):
        n = self.sample_size
        maxSpawnDistance = self.choose(self.maxSpawnDistances, n)
        maxAgentSpeed = self.choose(self.maxAgentSpeeds, n)
        maxReach = self.choose(self.maxReaches, n)
        maxBladeSpeed = self.choose(self.maxBladeSpeeds, n)
        agentPosition = maxSpawnDistance * self.runif(n) * self.rdir(n)
        agentVelocity = maxAgentSpeed * self.runif(n) * self.rdir(n)
        bladePosition = agentPosition + maxReach * self.runif(n) * self.rdir(n)
        bladeVeclocity = maxBladeSpeed * self.runif(n) * self.rdir(n)
        start = torch.cat((agentPosition, agentVelocity, bladePosition, bladeVeclocity),dim=1)
        return start
    
    def generate(self):
        agentStarts = [self.getAgentStart() for i in range(2)]
        start = torch.cat(agentStarts, dim=1)
        for i in range(2):
            agent = self.agents[i]
            agentState = agentStarts[i].repeat_interleave(81,dim=0)
            agent.position = agentState[:,0:2]
            agent.velocity = agentState[:,2:4]
            agent.blade.position = agentState[:,4:6]
            agent.blade.velocity = agentState[:,6:8]
        for _ in range(self.steps):
            self.step()
        agent0 = self.agents[0]
        agent1 = self.agents[1]
        blade0 = agent0.blade
        blade1 = agent1.blade
        outcomes = torch.cat([
            agent0.position,
            agent0.velocity,
            blade0.position,
            blade0.velocity,
            agent1.position,
            agent1.velocity,
            blade1.position,
            blade1.velocity
        ], dim=1).reshape(-1,81*16)
        data = torch.cat([start, outcomes],dim=1)
        return data

generator = Generator(10000, device)
data = generator.generate()
print(data.shape)