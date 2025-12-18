import { pi } from '../math'
import { World } from '../world/world'

export class Agent {
  static radius = 0.5
  static drag = 0.7
  static mass = pi * 0.25
  static movePower = 3
  world: World
  id: number
  position: number[]
  velocity = [0, 0]
  actionForce = [0, 0]
  collideForce = [0, 0]
  action = 0

  constructor (world: World, position = [0, 0]) {
    this.world = world
    this.position = position
    this.id = world.agents.length
    this.world.agents.push(this)
  }

  summarize (): AgentSummary {
    return {
      id: this.id,
      position: this.position
    }
  }
}

export interface AgentSummary {
  id: number
  position: number[]
}
