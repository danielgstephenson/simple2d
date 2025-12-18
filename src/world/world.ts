import { actionVectors } from '../actionVectors'
import { Agent, AgentSummary } from '../entities/agent'
import { Arena, ArenaSummary } from '../entities/arena'
import { Blade, BladeSummary } from '../entities/blade'
import { Wall, WallSummary } from '../entities/wall'
import { add, combine, dirFromTo, getDistance, mul } from '../math'

export class World {
  agents: Agent[] = []
  walls: Wall[] = []
  blades: Blade[] = []
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
    this.agents.forEach(agent => {
      agent.actionForce = mul(Agent.movePower, actionVectors[agent.action])
    })
    this.blades.forEach(blade => {
      const agent = this.agents[blade.id]
      const distance = getDistance(blade.position, agent.position)
      const dir = dirFromTo(blade.position, agent.position)
      blade.actionForce = mul(Blade.movePower * distance, dir)
    })
    this.blades.forEach(blade => {
      const force = add(blade.actionForce, blade.collideForce)
      blade.velocity = combine(1 - Blade.drag * dt, blade.velocity, dt / Agent.mass, force)
      blade.position = combine(1, blade.position, dt, blade.velocity)
    })
    this.agents.forEach(agent => {
      const force = add(agent.actionForce, agent.collideForce)
      agent.velocity = combine(1 - Agent.drag * dt, agent.velocity, dt / Agent.mass, force)
      agent.position = combine(1, agent.position, dt, agent.velocity)
    })
    this.summary = this.summarize()
  }
}

export interface WorldSummary {
  agents: AgentSummary[]
  blades: BladeSummary[]
  walls: WallSummary[]
  arena: ArenaSummary
}
