import { add, clamp, combine, mean, range, sub } from "../math"
import { World } from "../world/world"
import { Player } from "./agent/player"
import { Star } from "./star"

export class Door {
  spawnShell: number[][]
  shell: number[][]
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
    const xs = shell.map(p => p[0])
    const ys = shell.map(p => p[1])
    this.center = [mean(xs), mean(ys)]
    this.spawnCenter = structuredClone(this.center)
    this.openCenter = add(this.spawnCenter, vector)
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
      center: this.center
    }
  }
}

export interface DoorSummary {
  shell: number[][]
  center: number[]
}

