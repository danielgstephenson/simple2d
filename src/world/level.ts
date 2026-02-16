import { AgentSummary } from '../entities/circle/agent'
import { Player } from '../entities/circle/player'
import { BladeSummary } from '../entities/circle/blade'
import { DoorSummary } from '../entities/door'
import { Floor } from '../entities/floor'
import { RockSummary } from '../entities/circle/rock'
import { StarSummary } from '../entities/star'
import { TransporterSummary } from '../entities/transporter'
import { getChildById, getChildrenByRole, getCircleCenter, getCircleRadius, getPathPoints, parseSvg } from '../svg'
import { World } from './world'

export class Level extends World {
  player: Player
  floor: Floor
  summary: LevelSummary

  constructor() {
    super()
    const svgObject = parseSvg('test.svg')
    const layer1 = getChildById(svgObject, 'layer1')
    this.player = this.addPlayer(getCircleCenter(getChildById(layer1, 'player')))
    getChildrenByRole(layer1, 'boundary').forEach(node => {
      this.boundaries.push(getPathPoints(node))
    })
    getChildrenByRole(layer1, 'agent').forEach(node => {
      const position = getCircleCenter(node)
      this.addAgent(position)
    })
    getChildrenByRole(layer1, 'rock').forEach(node => {
      const position = getCircleCenter(node)
      const radius = getCircleRadius(node)
      this.addRock(position, radius)
    })
    getChildrenByRole(layer1, 'transporter').forEach(group => {
      console.log(group.attributes)
      const circle = getChildrenByRole(group, 'circle')[0]
      console.log(circle.attributes)
    })
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
      rocks: this.rocks.map(c => c.summarize()),
      agents: this.agents.map(c => c.summarize()),
      blades: this.blades.map(b => b.summarize()),
      stars: this.stars.map(s => s.summarize()),
      doors: this.doors.map(d => d.summarize()),
      transporters: this.transporters.map(t => t.summarize())
    }
  }

  layout(): Layout {
    return {
      boundaries: this.boundaries,
      walls: this.walls,
      floorPoints: this.floor.points
    }
  }
}

export interface LevelSummary {
  rocks: RockSummary[]
  agents: AgentSummary[]
  blades: BladeSummary[]
  stars: StarSummary[]
  doors: DoorSummary[]
  transporters: TransporterSummary[]
}

export interface Layout {
  boundaries: number[][][]
  walls: number[][][]
  floorPoints: number[][]
}