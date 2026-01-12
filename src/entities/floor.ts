import { getDistance, range, sortBy } from '../math'
import { insideWall, pointDistWall, segmentCastSegment } from '../raycast'
import { Level } from '../world/level'
import { Transporter } from './transporter'

export class Floor {
  level: Level
  basePoints: number[][] = []
  shells: number[][][] = []
  points: number[][] = []
  edges: number[][][] = []
  spread = 2

  constructor(level: Level) {
    this.level = level
    this.setupShells()
    this.setupBasePoints()
    this.setupPoints()
    this.setupEdges()
    console.log('this.edges.length', this.edges.length)
  }

  // Make this more efficient
  setupEdges(): void {
    const options = [...this.basePoints, ...this.points]
    for (const _ of range(10)) {
      for (const point of this.points) {
        const distances = options.map(option => getDistance(point, option))
        const sorted = sortBy(options, distances)
        sorted.pop()
        for (const otherPoint of sorted) {
          const edge = [point, otherPoint]
          const edgeLength = getDistance(edge[0], edge[1])
          if (edgeLength < this.spread) continue
          let invalid = false
          for (const otherEdge of this.edges) {
            const intersections = segmentCastSegment(edge, otherEdge)
            if (intersections.length > 0) {
              invalid = true
              break
            }
          }
          if (invalid) continue
          this.edges.push(edge)
          break
        }
      }
    }
  }

  setupPoints(): void {
    const xBoundary = this.level.boundary.map(point => point[0])
    const yBoundary = this.level.boundary.map(point => point[1])
    const xMax = Math.max(...xBoundary)
    const xMin = Math.min(...xBoundary)
    const yMax = Math.max(...yBoundary)
    const yMin = Math.min(...yBoundary)
    const xRange = xMax - xMin
    const yRange = yMax - yMin
    for (const _ of range(1000)) {
      const x = xMin + xRange * Math.random()
      const y = yMin + yRange * Math.random()
      const point = [x, y]
      if (!insideWall(point, this.level.boundary)) continue
      let invalid = false
      for (const wall of this.level.walls) {
        if (insideWall(point, wall)) {
          invalid = true
          break
        }
      }
      if (invalid) continue
      for (const transporter of this.level.transporters) {
        const distance = getDistance(point, transporter.center)
        if (distance < Transporter.radius) {
          invalid = true
          break
        }
      }
      if (invalid) continue
      for (const star of this.level.stars) {
        const distance = getDistance(point, star.spawnPoint)
        if (distance < this.spread) {
          invalid = true
          break
        }
      }
      if (invalid) continue
      for (const otherPoint of this.points) {
        const distance = getDistance(point, otherPoint)
        if (distance < this.spread) {
          invalid = true
          break
        }
      }
      if (invalid) continue
      for (const shell of this.shells) {
        const distance = pointDistWall(point, shell)
        if (distance < this.spread) {
          invalid = true
          break
        }
      }
      if (invalid) continue
      this.points.push(point)
    }
  }

  setupBasePoints(): void {
    this.basePoints.push(...this.level.boundary)
    this.level.walls.forEach(wall => {
      this.basePoints.push(...wall)
    })
    this.level.transporters.forEach(transporter => {
      this.basePoints.push(...transporter.shell)
    })
    this.level.stars.forEach(star => {
      this.basePoints.push(star.spawnPoint)
    })
  }

  setupShells(): void {
    this.shells.push(this.level.boundary)
    this.level.walls.forEach(wall => {
      this.shells.push(wall)
    })
    this.level.transporters.forEach(transporter => {
      this.shells.push(transporter.shell)
    })
  }
}
