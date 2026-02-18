import { add, clamp, combine, meanPoint, range, sub } from "../math"
import { World } from "../world/world"
import { Player } from "./circle/player"
import { Star } from "./star"

export class Door {
  spawnShell: number[][]
  shell: number[][]
  knots: number[][] = []
  spawnCenter: number[]
  center: number[]
  openCenter: number[]
  world: World
  index: number
  star?: Star
  open = false
  time = 0
  moveInterval = 4

  constructor(world: World, vector: number[], shell: number[][]) {
    this.world = world
    this.spawnShell = structuredClone(shell)
    this.shell = structuredClone(shell)
    this.index = world.doors.length
    this.world.doors.push(this)
    this.center = meanPoint(shell)
    this.spawnCenter = structuredClone(this.center)
    this.openCenter = add(this.spawnCenter, vector)
    this.knots.push([0, 0])
    // const xs = shell.map(p => p[0])
    // const ys = shell.map(p => p[1])
    // const xMax = Math.max(...xs)
    // const xMin = Math.min(...xs)
    // const yMax = Math.max(...ys)
    // const yMin = Math.min(...ys)
    // const xRange = xMax - xMin
    // const yRange = yMax - yMin
    // const count = Math.ceil(xRange * yRange / 5)
    // for (const _ of range(count)) {
    //   const x = xRange * (Math.random() - 0.5)
    //   const y = yRange * (Math.random() - 0.5)
    //   const knot = [x, y]
    //   this.knots.push(knot)
    // }
  }

  onStep(dt: number): void {
    if (!this.open) return
    this.time += dt
    const factor = clamp(0, 1, this.time / this.moveInterval)
    this.center = combine(factor, this.openCenter, 1 - factor, this.spawnCenter)
    const offset = sub(this.center, this.spawnCenter)
    range(this.shell.length).forEach(i => {
      this.shell[i] = add(this.spawnShell[i], offset)
    })
  }

  knock(player: Player): void {
    if (this.star != null) return
    if (player.star == null) return
    this.open = true
    this.star = player.star
    player.star = undefined
    this.star.agent = undefined
    this.star.door = this
  }

  summarize(): DoorSummary {
    return {
      shell: this.shell,
      center: this.center,
      knots: this.knots
    }
  }
}

export interface DoorSummary {
  shell: number[][]
  center: number[]
  knots: number[][]
}

