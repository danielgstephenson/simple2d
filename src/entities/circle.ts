import { pi } from '../math'
import { World } from '../world/world'

export class Circle {
  world: World
  radius: number
  mass: number
  position: number[]
  id: number
  velocity = [0, 0]
  actionForce = [0, 0]
  collideForce = [0, 0]
  drag = 0

  constructor (world: World, position = [0, 0], radius = 0.5) {
    this.world = world
    this.radius = radius
    this.mass = pi * this.radius ** 2
    this.position = position
    this.id = world.circles.length
    this.world.circles.push(this)
  }
}

export interface CircleSummary {
  position: number[]
}
