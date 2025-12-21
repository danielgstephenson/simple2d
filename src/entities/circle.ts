import { pi } from '../math'
import { World } from '../world/world'

export class Circle {
  static historyLength = 15
  world: World
  radius: number
  mass: number
  position: number[]
  id: number
  history: number[][] = []
  velocity = [0, 0]
  force = [0, 0]
  impulse = [0, 0]
  shift = [0, 0]
  drag = 0

  constructor (world: World, position = [0, 0], radius = 0.5) {
    this.world = world
    this.radius = radius
    this.mass = pi * radius ** 2
    this.position = position
    this.id = world.circles.length
    this.world.circles.push(this)
  }

  summarize (): CircleSummary {
    return {
      position: this.position,
      history: this.history
    }
  }
}

export interface CircleSummary {
  position: number[]
  history: number[][]
}
