import { actionVectors } from '../actionVectors'
import { Circle, CircleSummary } from '../entities/circle'
import { Wall, WallSummary } from '../entities/wall'
import { combine, mul } from '../math'

export class World {
  circles: Circle[] = []
  walls: Wall[] = []
  summary: WorldSummary
  timeStep = 0.04
  timeScale = 1

  constructor () {
    this.summary = this.summarize()
    setInterval(() => this.step(), 1000 * this.timeStep / this.timeScale)
  }

  addCircle (position: number[], radius = 0.5): Circle {
    return new Circle(this, position, radius)
  }

  addWall (a: number[], b: number[]): Wall {
    return new Wall(this, a, b)
  }

  summarize (): WorldSummary {
    return {
      circles: this.circles.map(c => c.summarize()),
      walls: this.walls.map(w => w.summarize())
    }
  }

  step (): void {
    const dt = this.timeStep
    this.circles.forEach(circle => {
      circle.force = mul(5, actionVectors[circle.action])
      circle.velocity = combine(1 - circle.drag * dt, circle.velocity, dt / circle.mass, circle.force)
      circle.position = combine(1, circle.position, dt, circle.velocity)
    })
    this.summary = this.summarize()
  }
}

export interface WorldSummary {
  circles: CircleSummary[]
  walls: WallSummary[]
}
