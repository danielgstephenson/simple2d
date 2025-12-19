import { World } from '../world/world'
import { Circle } from './circle'

export class Blade extends Circle {
  static radius = 1
  drag = 0.3
  movePower = 4
  velocity = [0, 0]
  force = [0, 0]
  collideForce = [0, 0]

  constructor (world: World, position = [0, 0]) {
    super(world, position, Blade.radius)
    this.world.blades.push(this)
  }
}
