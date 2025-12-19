import { World } from '../world/world'
import { Circle, CircleSummary } from './circle'

export class Agent extends Circle {
  static radius = 0.5
  static drag = 0.7
  static movePower = 3
  velocity = [0, 0]
  actionForce = [0, 0]
  collideForce = [0, 0]
  action = 0

  constructor (world: World, position = [0, 0]) {
    super(world, position, 0.5)
    this.world.agents.push(this)
  }

  summarize (): CircleSummary {
    return {
      position: this.position
    }
  }
}
