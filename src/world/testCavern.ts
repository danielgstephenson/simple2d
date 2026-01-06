import { Agent } from '../entities/agent'
import { World } from './world'

export class TestCavern extends World {
  player: Agent
  timeScale = 1.4
  timeStep = 0.02

  constructor () {
    super()
    this.player = this.addAgent([-3, 0])
    this.boundary = [
      [0, -20],
      [20, 20],
      [-20, 20]
    ]
    this.summary = this.summarize()
    this.begin()
  }

  postStep (): void {
    this.summary = this.summarize()
    if (this.player.dead) {
      this.paused = true
    }
  }
}
