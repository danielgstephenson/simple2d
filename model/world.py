from __future__ import annotations
from math import cos, pi, sin
import torch
import torch.nn.functional as F

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# defaultType = torch.float64
torch.set_printoptions(sci_mode=False)

# actionVectorList = [[0.0,0.0]]
# for i in range(8):
#     angle = 2 * pi * i / 8
#     dir = [cos(angle), sin(angle)]
#     actionVectorList.append(dir)
# actionVectors = torch.tensor(actionVectorList).to(device)
# actions = torch.tensor([i for i in range(9)]).to(device)
# actionIndexPairs = torch.cartesian_prod(actions,actions)
# actionVectorPairs = torch.cat(
#     [
#         actionVectors[actionIndexPairs[:,0]],
#         actionVectors[actionIndexPairs[:,1]]
#     ],
#     dim=1
# )

class Circle: 
    def __init__(self,world: World, radius: float):
        self.world = world
        self.radius = radius
        self.mass = pi * radius ** 2
        self.drag = 0
        self.position = torch.zeros(world.count,2,dtype=world.dtype).to(world.device)
        self.velocity = torch.zeros(world.count,2,dtype=world.dtype).to(world.device)
        self.force = torch.zeros(world.count,2,dtype=world.dtype).to(world.device)
        self.impulse = torch.zeros(world.count,2,dtype=world.dtype).to(world.device)
        self.shift = torch.zeros(world.count,2,dtype=world.dtype).to(world.device)
        self.id = len(self.world.circles)
        self.world.circles.append(self)

class Agent(Circle):
    def __init__(self, world: World):
        super().__init__(world, 0.5)
        self.blade = Blade(self)
        self.movePower = 3
        self.drag = 0.7
        self.dead = torch.zeros(world.count, dtype=torch.int).to(world.device)
        self.action = torch.zeros(world.count, dtype=torch.int).to(world.device)
        self.world.agents.append(self)

class Blade(Circle):
    def __init__(self, agent: Agent):
        super().__init__(agent.world, 1)
        self.agent = agent
        self.movePower = 4
        self.drag = 0.3
        self.world.blades.append(self)

class World:
    def __init__(self, count: int, device: torch.device, dtype = torch.float64):
        self.timeStep = 0.04
        self.circles: list[Circle] = []
        self.agents: list[Agent] = []
        self.blades: list[Blade] = []
        self.count = count
        self.device = device
        self.dtype = dtype
        actionVectorList = [[0.0,0.0]]
        for i in range(8):
            angle = 2 * pi * i / 8
            dir = [cos(angle), sin(angle)]
            actionVectorList.append(dir)
        self.actionVectors = torch.tensor(actionVectorList).to(device)
        self.actions = torch.tensor([i for i in range(9)]).to(device)

    def checkDeath(self, agent: Agent):
        for blade in self.blades:
            if blade.agent.id == agent.id: continue
            vector = agent.position - blade.position
            distance = torch.linalg.norm(vector, dim=1)
            overlap = agent.radius + blade.radius - distance
            agent.dead = torch.where(overlap > 0, 1, agent.dead)

    def collideCircleCircle(self, circle1: Circle, circle2: Circle):
        if circle1.id >= circle2.id: return
        vector = circle2.position - circle1.position
        distance = torch.linalg.norm(vector, dim=1)
        overlap = (circle1.radius + circle2.radius - distance).unsqueeze(1)
        normal = F.normalize(vector, dim=1)
        relativeVelocity = circle1.velocity - circle2.velocity
        impactSpeed = torch.linalg.vecdot(relativeVelocity, normal).unsqueeze(1)
        massFactor = 1 / circle1.mass + 1 / circle2.mass
        impulse = torch.where(overlap > 0, impactSpeed / massFactor * normal, 0)
        shift = torch.where(overlap > 0, 0.5 * overlap * normal, 0)
        circle1.impulse = circle1.impulse - impulse
        circle2.impulse = circle2.impulse + impulse
        circle1.shift = circle1.shift - shift
        circle2.shift = circle2.shift + shift
            
    def step(self):
        dt = self.timeStep
        agentCount = len(self.agents)
        for i in range(agentCount):
            agent = self.agents[i]
            blade = self.blades[i]
            agent.force[:,:] = 0
            blade.force[:,:] = 0
            agent.impulse[:,:] = 0
            blade.impulse[:,:] = 0
            agent.shift[:,:] = 0
            blade.shift[:,:] = 0
        for agent in self.agents:
            agent.force = agent.movePower * self.actionVectors[agent.action]
        for blade in self.blades:
            vector = blade.agent.position - blade.position
            blade.force = blade.movePower * vector
        for agent1 in self.agents:
            for agent2 in self.agents:
                self.collideCircleCircle(agent1, agent2)
        for blade1 in self.blades:
            for blade2 in self.blades:
                self.collideCircleCircle(blade1, blade2)
        for circle in self.circles:
            circle.velocity = (1 - circle.drag * dt) * circle.velocity 
            circle.velocity = circle.velocity + dt / circle.mass * circle.force
            circle.velocity = circle.velocity + 1 / circle.mass * circle.impulse
            circle.position = circle.position + dt * circle.velocity + circle.shift
        for agent in self.agents:
            self.checkDeath(agent)
        for agent in self.agents:
            direction = F.normalize(agent.position, dim=1)
            dead = (agent.dead > 0).unsqueeze(1)
            agent.position = torch.where(dead, agent.position + 15 * direction, agent.position)
            agent.blade.position = torch.where(dead, agent.position.clone(), agent.blade.position)
            agent.velocity =  torch.where(dead, 0*agent.velocity, agent.velocity)
            agent.blade.velocity = torch.where(dead, 0*agent.blade.velocity, agent.blade.velocity)
            agent.dead = 0 * agent.dead




