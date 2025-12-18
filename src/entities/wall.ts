import { World } from '../world/world'

export class Wall {
  world: World
  id: number
  a: number[]
  b: number[]

  constructor (world: World, a: number[], b: number[]) {
    this.world = world
    this.a = a
    this.b = b
    this.id = world.walls.length
    this.world.walls.push(this)
  }

  summarize (): WallSummary {
    return {
      id: this.id,
      a: this.a,
      b: this.b
    }
  }
}

export interface WallSummary {
  id: number
  a: number[]
  b: number[]
}
