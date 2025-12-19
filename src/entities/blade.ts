import { pi } from '../math'
import { World } from '../world/world'
import { Circle, CircleSummary } from './circle'

export class Blade extends Circle {
  static radius = 1
  static drag = 0.3
  static mass = pi
  static movePower = 4
  velocity = [0, 0]
  actionForce = [0, 0]
  collideForce = [0, 0]

  constructor (world: World, position = [0, 0]) {
    super(world, position, Blade.radius)
    this.world.blades.push(this)
  }

  summarize (): CircleSummary {
    return {
      position: this.position
    }
  }
}
