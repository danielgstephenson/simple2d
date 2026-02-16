import { angleToDir, combine, pi, range } from '../math'
import { World } from '../world/world'
import { Agent } from './circle/agent'

export class Transporter {
  static radius = 4
  world: World
  charge = 0
  interval = 5
  center: number[]
  target: number[]
  shell: number[][]
  index: number

  constructor(world: World, center: number[], target: number[]) {
    this.world = world
    this.center = center
    this.target = target
    this.index = world.transporters.length
    this.world.transporters.push(this)
    this.shell = range(8).map(i => {
      const angle = i / 8 * 2 * pi
      const dir = angleToDir(angle)
      return combine(1, this.center, Transporter.radius, dir)
    })
  }

  transport(agent: Agent) {
    this.charge = 0
    if (agent.blade != null) agent.blade.detach()
    agent.position = this.target
  }

  summarize(): TransporterSummary {
    return {
      center: this.center,
      charge: this.charge,
      interval: this.interval
    }
  }
}

export interface TransporterSummary {
  center: number[]
  charge: number
  interval: number
}
