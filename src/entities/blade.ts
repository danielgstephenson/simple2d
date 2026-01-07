import { World } from '../world/world'
import { Agent } from './agent'
import { Circle } from './circle'

export class Blade extends Circle {
  static radius = 1
  agent?: Agent
  drag = 0.3
  movePower = 4
  velocity = [0, 0]
  force = [0, 0]
  collideForce = [0, 0]
  align = 0
  index: number

  constructor (world: World, position = [0, 0]) {
    super(world, position, Blade.radius)
    this.index = this.world.blades.length
    this.world.blades.push(this)
  }

  attach (agent: Agent): void {
    agent.blade = this
    this.agent = agent
    this.align = agent.align
  }

  detach (): void {
    if (this.agent != null) {
      this.agent.blade = undefined
      this.agent = undefined
    }
  }

  summarize (): BladeSummary {
    return {
      position: this.position,
      history: this.history,
      align: this.align,
      agent: this.agent == null ? undefined : this.agent.index
    }
  }
}

export interface BladeSummary {
  position: number[]
  history: number[][]
  align: number
  agent?: number
}
