import { World } from '../world/world'
import { Agent } from './circle/agent'
import { Door } from './door'

export class Star {
  static radius = 0.4
  world: World
  spawnPoint = [0, 0]
  agent?: Agent
  door?: Door
  index: number

  constructor(world: World, position = [0, 0]) {
    this.world = world
    this.spawnPoint = position
    this.index = world.stars.length
    this.world.stars.push(this)
  }

  reset(): void {
    if (this.agent != null) this.agent.star = undefined
    this.agent = undefined
    if (this.door != null) {
      this.door.star = undefined
    }
    this.door = undefined
  }

  summarize(): StarSummary {
    return {
      spawnPoint: this.spawnPoint,
      agent: this.agent == null ? undefined : this.agent.index,
      door: this.door == null ? undefined : this.door.index,
    }
  }
}

export interface StarSummary {
  spawnPoint: number[]
  agent?: number
  door?: number
}
