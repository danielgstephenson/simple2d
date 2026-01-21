import { mean } from "../math"
import { World } from "../world/world"
import { Player } from "./agent/player"
import { Star } from "./star"

export class Door {
  spawnShell: number[][]
  shell: number[][]
  center: number[]
  vector: number[]
  world: World
  index: number
  star?: Star
  open = false
  openTime = 2

  constructor(world: World, vector: number[], shell: number[][]) {
    this.world = world
    this.spawnShell = structuredClone(shell)
    this.shell = structuredClone(shell)
    this.vector = structuredClone(vector)
    this.index = world.doors.length
    this.world.doors.push(this)
    const xs = shell.map(p => p[0])
    const ys = shell.map(p => p[1])
    this.center = [mean(xs), mean(ys)]
  }

  knock(player: Player): void {
    console.log('knock')
    if (this.star != null) return
    if (player.star == null) return
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

