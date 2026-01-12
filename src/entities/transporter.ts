import { angleToDir, combine, pi, range } from '../math'
import { World } from '../world/world'

export class Transporter {
  static radius = 5
  world: World
  center = [0, 0]
  shell: number[][]
  index: number

  constructor(world: World, position = [0, 0]) {
    this.world = world
    this.center = position
    this.index = world.transporters.length
    this.world.transporters.push(this)
    this.shell = range(8).map(i => {
      const angle = i / 8 * 2 * pi
      const dir = angleToDir(angle)
      return combine(1, this.center, Transporter.radius, dir)
    })
  }

  summarize(): TransporterSummary {
    return {
      center: this.center
    }
  }
}

export interface TransporterSummary {
  center: number[]
}
