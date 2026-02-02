import { AgentSummary } from '../entities/agent/agent'
import { Player } from '../entities/agent/player'
import { BladeSummary } from '../entities/blade'
import { DoorSummary } from '../entities/door'
import { Floor } from '../entities/floor'
import { StarSummary } from '../entities/star'
import { TransporterSummary } from '../entities/transporter'
import { getChildById, getCircleCenter, getPathPoints, parseSvg } from '../svg'
import { World } from './world'

export class Level extends World {
  player: Player
  floor: Floor
  summary: LevelSummary

  constructor() {
    super()
    const svgObject = parseSvg('test.svg')
    const layer1 = getChildById(svgObject, 'layer1')
    // console.log(JSON.stringify(layer1, null, 2))
    this.player = this.addPlayer(getCircleCenter(getChildById(layer1, 'player')))
    this.boundary = getPathPoints(getChildById(layer1, 'boundary'))
    this.floor = new Floor(this)
    this.summary = this.summarize()
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