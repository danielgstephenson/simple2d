import { pi } from '../math'
import { World } from '../world/world'

export class Circle {
  world: World
  id: number
  radius: number
  position: number[]
  velocity: number[]
  force: number[]
  mass = 1
  drag = 0.7
  action = 0

  constructor (world: World, position = [0, 0], radius = 0.5) {
    this.world = world
    this.position = position
    this.radius = radius
    this.mass = pi * this.radius ** 2
    this.velocity = [0, 0]
    this.force = [0, 0]
    this.id = world.circles.length
    this.world.circles.push(this)
  }

  summarize (): CircleSummary {
    return {
      id: this.id,
      radius: this.radius,
      position: this.position
    }
  }
}

export interface CircleSummary {
  id: number
  radius: number
  position: number[]
}
