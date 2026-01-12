import { Agent, AgentSummary } from '../entities/agent/agent'
import { Player } from '../entities/agent/player'
import { BladeSummary } from '../entities/blade'
import { Floor } from '../entities/floor'
import { Star, StarSummary } from '../entities/star'
import { TransporterSummary } from '../entities/transporter'
import { getDistance } from '../math'
import { World } from './world'

export class Level extends World {
  player: Player
  timeScale = 1.4
  timeStep = 0.02
  floor: Floor
  summary: LevelSummary

  constructor() {
    super()
    this.player = this.addPlayer([-3, 0])
    this.player.align = 1
    this.addPlayerBlade([0, 10])
    this.addGuard([10, 0])
    this.addGuard([-10, 0])
    this.addGuardBlade([-10, 0])
    this.addStar([0, -10])
    this.addTransporter([0, -1])
    this.boundary = [
      [-10, -30],
      [10, -20],
      [25, 25],
      [-25, 15]
    ]
    this.walls.push([
      [5, 5],
      [5, 18],
      [18, 18]
    ])
    this.floor = new Floor(this)
    this.summary = this.summarize()
    this.begin()
  }

  postStep(): void {
    this.summary = this.summarize()
    if (this.player.dead) {
      this.paused = true
    }
  }

  preStep(): void {
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

  summarize(): LevelSummary {
    return {
      boundary: this.boundary,
      walls: this.walls,
      agents: this.agents.map(c => c.summarize()),
      blades: this.blades.map(b => b.summarize()),
      stars: this.stars.map(s => s.summarize()),
      transporters: this.transporters.map(t => t.summarize()),
      floorPoints: this.floor.points,
      floorEdges: this.floor.edges
    }
  }
}

export interface LevelSummary {
  agents: AgentSummary[]
  blades: BladeSummary[]
  stars: StarSummary[]
  transporters: TransporterSummary[]
  walls: number[][][]
  boundary: number[][]
  floorPoints: number[][]
  floorEdges: number[][][]
}