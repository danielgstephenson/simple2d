import { pi } from '../math'
import { World } from '../world/world'

export class Blade {
  static radius = 1
  static drag = 0.3
  static mass = pi
  static movePower = 2
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
    this.id = world.blades.length
    this.world.blades.push(this)
  }

  summarize (): BladeSummary {
    return {
      id: this.id,
      position: this.position
    }
  }
}

export interface BladeSummary {
  id: number
  position: number[]
}
