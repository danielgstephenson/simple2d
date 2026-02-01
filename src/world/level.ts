import { AgentSummary } from '../entities/agent/agent'
import { Player } from '../entities/agent/player'
import { BladeSummary } from '../entities/blade'
import { DoorSummary } from '../entities/door'
import { Floor } from '../entities/floor'
import { StarSummary } from '../entities/star'
import { TransporterSummary } from '../entities/transporter'
import { World } from './world'

export class Level extends World {
  player: Player
  timeScale = 1
  timeStep = 0.02
  floor: Floor
  summary: LevelSummary

  constructor() {
    super()
    this.player = this.addPlayer([-3, 0])
    this.boundary = [
      [-10, -30],
      [10, -10],
      [25, 35],
      [-25, 20]
    ]
    this.walls.push([
      [30, -10],
      [-30, 40],
      [10, 10]
    ])
    this.addDoor([0, -5], [
      [-10, 10],
      [-10, 13],
      [-14, 13],
      [-14, 10]
    ])
    this.addTransporter([4, 4], [13, 13])
    this.addTransporter([2, 23], [-3, 13])
    this.addPlayerBlade([0, 10])
    this.addGuard([10, 0])
    this.addGuard([-10, 0])
    this.addGuardBlade([-10, 0])
    this.addStar([0, -10])
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

  summarize(): LevelSummary {
    return {
      agents: this.agents.map(c => c.summarize()),
      blades: this.blades.map(b => b.summarize()),
      stars: this.stars.map(s => s.summarize()),
      doors: this.doors.map(d => d.summarize()),
      transporters: this.transporters.map(t => t.summarize())
    }
  }

  layout(): Layout {
    return {
      boundary: this.boundary,
      walls: this.walls,
      floorPoints: this.floor.points
    }
  }
}

export interface LevelSummary {
  agents: AgentSummary[]
  blades: BladeSummary[]
  stars: StarSummary[]
  doors: DoorSummary[]
  transporters: TransporterSummary[]
}

export interface Layout {
  boundary: number[][]
  walls: number[][][]
  floorPoints: number[][]
}