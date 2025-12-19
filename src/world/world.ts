import { actionVectors } from '../actionVectors'
import { Agent } from '../entities/agent'
import { Arena, ArenaSummary } from '../entities/arena'
import { Blade } from '../entities/blade'
import { Circle, CircleSummary } from '../entities/circle'
import { Wall, WallSummary } from '../entities/wall'
import { combine, dirFromTo, dot, getDistance, mul, range, sub } from '../math'

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
    const agent = new Agent(this, position)
    return agent
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
      agent.force = [0, 0]
      blade.force = [0, 0]
      agent.impulse = [0, 0]
      blade.impulse = [0, 0]
      agent.shift = [0, 0]
      blade.shift = [0, 0]
    })
    range(agentCount).forEach(i => {
      const agent = this.agents[i]
      agent.force = mul(agent.movePower, actionVectors[agent.action])
    })
    range(agentCount).forEach(i => {
      const blade = this.blades[i]
      const agent = this.agents[i]
      const distance = getDistance(blade.position, agent.position)
      const dir = dirFromTo(blade.position, agent.position)
      blade.force = mul(blade.movePower * distance, dir)
    })
    range(agentCount).forEach(i => {
      range(agentCount).forEach(j => {
        if (i >= j) return
        const agent1 = this.agents[i]
        const agent2 = this.agents[j]
        this.collideCircleCircle(agent1, agent2)
        const blade1 = this.blades[i]
        const blade2 = this.blades[j]
        this.collideCircleCircle(blade1, blade2)
      })
    })
    this.circles.forEach(circle => {
      circle.velocity = mul(1 - circle.drag * dt, circle.velocity)
      circle.velocity = combine(1, circle.velocity, dt / circle.mass, circle.force)
      circle.velocity = combine(1, circle.velocity, 1 / circle.mass, circle.impulse)
      circle.position = combine(1, circle.position, dt, circle.velocity)
      circle.position = combine(1, circle.position, 1, circle.shift)
    })
    this.agents.forEach(agent => this.checkDeath(agent))
    this.summary = this.summarize()
  }

  collideCircleCircle (circle1: Circle, circle2: Circle): void {
    if (circle1.id >= circle2.id) return
    const distance = getDistance(circle1.position, circle2.position)
    const overlap = circle1.radius + circle2.radius - distance
    if (overlap <= 0) return
    const normal = dirFromTo(circle1.position, circle2.position)
    const relativeVelocity = sub(circle1.velocity, circle2.velocity)
    const impactSpeed = dot(relativeVelocity, normal)
    const massFactor = 1 / circle1.mass + 1 / circle2.mass
    const impulse = mul(impactSpeed / massFactor, normal)
    const shift = mul(0.5 * overlap, normal)
    circle1.impulse = combine(1, circle1.impulse, -1, impulse)
    circle2.impulse = combine(1, circle2.impulse, +1, impulse)
    circle1.shift = combine(1, circle1.shift, -1, shift)
    circle2.shift = combine(1, circle2.shift, +1, shift)
  }

  checkDeath (agent: Agent): void {
    this.blades.forEach(blade => {
      if (blade.id === agent.blade.id) return
      const distance = getDistance(agent.position, blade.position)
      const overlap = agent.radius + blade.radius - distance
      if (overlap < 0) return
      agent.die()
    })
  }
}

export interface WorldSummary {
  agents: CircleSummary[]
  blades: CircleSummary[]
  walls: WallSummary[]
  arena: ArenaSummary
}
