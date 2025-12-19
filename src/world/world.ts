import { actionVectors } from '../actionVectors'
import { Agent } from '../entities/agent'
import { Arena, ArenaSummary } from '../entities/arena'
import { Blade } from '../entities/blade'
import { Circle, CircleSummary } from '../entities/circle'
import { Wall, WallSummary } from '../entities/wall'
import { add, combine, dirFromTo, getDistance, mul, range } from '../math'

export class World {
  agents: Agent[] = []
  walls: Wall[] = []
  blades: Blade[] = []
  circles: Circle[] = []
  arena: Arena
  summary: WorldSummary
  timeStep = 0.04
  timeScale = 1

  constructor () {
    this.arena = new Arena(this)
    this.summary = this.summarize()
    setInterval(() => this.step(), 1000 * this.timeStep / this.timeScale)
  }

  addAgent (position: number[]): Agent {
    void new Blade(this, position)
    return new Agent(this, position)
  }

  addWall (a: number[], b: number[]): Wall {
    return new Wall(this, a, b)
  }

  summarize (): WorldSummary {
    return {
      agents: this.agents.map(c => c.summarize()),
      blades: this.blades.map(b => b.summarize()),
      walls: this.walls.map(w => w.summarize()),
      arena: this.arena.summarize()
    }
  }

  step (): void {
    const dt = this.timeStep
    const agentCount = this.agents.length
    range(agentCount).forEach(i => {
      const agent = this.agents[i]
      const blade = this.blades[i]
      agent.collideForce = [0, 0]
      blade.collideForce = [0, 0]
    })
    range(agentCount).forEach(i => {
      const agent = this.agents[i]
      agent.actionForce = mul(Agent.movePower, actionVectors[agent.action])
    })
    range(agentCount).forEach(i => {
      const blade = this.blades[i]
      const agent = this.agents[i]
      const distance = getDistance(blade.position, agent.position)
      const dir = dirFromTo(blade.position, agent.position)
      blade.actionForce = mul(Blade.movePower * distance, dir)
    })
    range(agentCount).forEach(i => {
      range(agentCount).forEach(j => {
        if (i >= j) return
        const agent = this.agents[i]
        const otherAgent = this.agents[j]
        this.collideCircleCircle(agent, otherAgent)
        const blade = this.blades[i]
        const otherBlade = this.blades[j]
        this.collideCircleCircle(blade, otherBlade)
      })
    })
    this.blades.forEach(blade => {
      const force = add(blade.actionForce, blade.collideForce)
      blade.velocity = combine(1 - Blade.drag * dt, blade.velocity, dt / blade.mass, force)
      blade.position = combine(1, blade.position, dt, blade.velocity)
    })
    this.agents.forEach(agent => {
      const force = add(agent.actionForce, agent.collideForce)
      agent.velocity = combine(1 - Agent.drag * dt, agent.velocity, dt / agent.mass, force)
      agent.position = combine(1, agent.position, dt, agent.velocity)
    })
    this.summary = this.summarize()
  }

  collideCircleCircle (circle1: Circle, circle2: Circle): void {
    if (circle1.id >= circle2.id) return
    const dt = this.timeStep
    const agentFuturePos = combine(1, circle1.position, dt, circle1.velocity)
    const otherFuturePos = combine(1, circle2.position, dt, circle2.velocity)
    const distance = getDistance(agentFuturePos, otherFuturePos)
    const overlap = circle1.radius + circle2.radius - distance
    if (overlap <= 0) return
    const intensity = 1
    const depthFactor = 100
    const power = intensity * depthFactor * Math.log(1 + Math.exp(-overlap / depthFactor))
    const normal = dirFromTo(circle1.position, circle2.position)
    const otherForce = mul(+power, normal)
    const agentForce = mul(-power, normal)
    circle2.collideForce = add(circle2.actionForce, otherForce)
    circle1.collideForce = add(circle1.actionForce, agentForce)
  }
}

export interface WorldSummary {
  agents: CircleSummary[]
  blades: CircleSummary[]
  walls: WallSummary[]
  arena: ArenaSummary
}
