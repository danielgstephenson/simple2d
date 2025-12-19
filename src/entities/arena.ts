import { World } from '../world/world'

export class Arena {
  static size = 100000
  world: World
  boundary: number[][]

  constructor (world: World) {
    this.world = world
    this.boundary = [
      [-Arena.size, -Arena.size],
      [-Arena.size, +Arena.size],
      [+Arena.size, +Arena.size],
      [+Arena.size, -Arena.size]
    ]
  }

  summarize (): ArenaSummary {
    return {
      boundary: this.boundary
    }
  }
}

export interface ArenaSummary {
  boundary: number[][]
}
