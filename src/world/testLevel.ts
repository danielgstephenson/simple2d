import { Agent } from '../entities/agent/agent'
import { Player } from '../entities/agent/player'
import { Star } from '../entities/star'
import { getDistance } from '../math'
import { World } from './world'

export class TestCavern extends World {
  player: Player
  timeScale = 1.4
  timeStep = 0.02

  constructor () {
    super()
    this.player = this.addPlayer([-3, 0])
    this.player.align = 1
    this.addBlade([0, 10])
    this.addAgent([10, 0])
    const guard = this.addGuard([-10, 0])
    guard.align = 2
    this.addBlade([-10.5, 0])
    this.addStar([0, -10])
    this.boundary = [
      [-10, -30],
      [10, -20],
      [25, 25],
      [-25, 15]
    ]
    this.walls.push([
      [5, 5],
      [5, 10],
      [10, 10]
    ])
    this.summary = this.summarize()
    this.begin()
  }

  postStep (): void {
    this.summary = this.summarize()
    if (this.player.dead) {
      this.paused = true
    }
  }

  preStep (): void {
    this.stars.forEach(star => {
      if (star.agent != null) return
      if (this.player.star != null) return
      const distance = getDistance(this.player.position, star.spawnPoint)
      if (distance > Star.radius + Agent.radius) return
      this.player.takeStar(star)
    })
    if (this.player.dead) {
      this.player.respawn()
    }
  }
}
