import { World } from '../world/world'
import { Agent } from './agent/agent'

export class Star {
  static radius = 0.4
  world: World
  spawnPoint = [0, 0]
  agent?: Agent
  index: number

  constructor (world: World, position = [0, 0]) {
    this.world = world
    this.spawnPoint = position
    this.index = world.stars.length
    this.world.stars.push(this)
  }

  reset (): void {
    if (this.agent != null) this.agent.star = undefined
    this.agent = undefined
  }

  summarize (): StarSummary {
    return {
      spawnPoint: this.spawnPoint,
      agent: this.agent == null ? undefined : this.agent.index
    }
  }
}

export interface StarSummary {
  spawnPoint: number[]
  agent?: number
}
