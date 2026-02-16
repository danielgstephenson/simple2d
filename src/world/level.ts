import { AgentSummary } from '../entities/circle/agent'
import { Player } from '../entities/circle/player'
import { BladeSummary } from '../entities/circle/blade'
import { DoorSummary } from '../entities/door'
import { Floor } from '../entities/floor'
import { RockSummary } from '../entities/circle/rock'
import { StarSummary } from '../entities/star'
import { Transporter, TransporterSummary } from '../entities/transporter'
import { World } from './world'
import { findElement, getCenter, getPathPoints, getRadius, getSvg } from '../svg'
import { getDistance } from '../math'

export class Level extends World {
  floor: Floor
  summary: LevelSummary

  constructor() {
    super()
    const svg = getSvg('test.svg')
    const playerElement = findElement(svg, '[role="player"]')
    this.addPlayer(getCenter(playerElement))
    svg.find('[role="boundary"]').forEach(element => {
      this.boundaries.push(getPathPoints(element))
    })
    svg.find('[role="agent"]').forEach(element => {
      this.addAgent(getCenter(element))
    })
    svg.find('[role="rock"]').forEach(element => {
      const position = getCenter(element)
      const radius = getRadius(element)
      this.addRock(position, radius)
    })
    const arrows = [...svg.find('[role="arrow"]')].map(element => {
      const points = getPathPoints(element)
      console.log('points', points)
      return points
    })
    console.log('arrows', arrows)
    svg.find('[role="transporter"]').forEach(element => {
      const position = getCenter(element)
      const arrow = arrows.find(points => {
        const start = points[0]
        console.log('start', start)
        const distance = getDistance(start, position)
        return distance <= Transporter.radius
      })
      if (arrow == null) throw new Error('arrow not found')
      const target = arrow[1]
      this.addTransporter(position, target)
    })
    this.floor = new Floor(this)
    this.summary = this.summarize()
  }


  postStep(): void {
    this.summary = this.summarize()
    this.players.forEach(player => {
      if (player.dead) this.paused = true
    })
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