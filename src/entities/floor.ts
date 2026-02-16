import { range } from '../math'
import { Level } from '../world/level'

export class Floor {
  points: number[][] = []

  constructor(level: Level) {
    if (level.boundaries.length == 0) return
    const xBoundary = level.boundaries.flat().map(point => point[0])
    const yBoundary = level.boundaries.flat().map(point => point[1])
    const xMax = Math.max(...xBoundary)
    const xMin = Math.min(...xBoundary)
    const yMax = Math.max(...yBoundary)
    const yMin = Math.min(...yBoundary)
    const xRange = xMax - xMin
    const yRange = yMax - yMin
    const count = Math.ceil(xRange * yRange / 5)
    for (const _ of range(count)) {
      const x = xMin + xRange * Math.random()
      const y = yMin + yRange * Math.random()
      const point = [x, y]
      this.points.push(point)
    }
  }
}
