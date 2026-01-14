import { World } from "../world/world"

export class Door {
  spawnShell: number[][]
  shell: number[][]
  vector: number[]
  world: World
  open = false
  openTime = 2

  constructor(world: World, vector: number[], shell: number[][]) {
    this.world = world
    this.spawnShell = structuredClone(shell)
    this.shell = structuredClone(shell)
    this.vector = structuredClone(vector)
    this.world.doors.push(this)
  }

  summarize(): DoorSummary {
    return {
      shell: this.shell
    }
  }
}

export interface DoorSummary {
  shell: number[][]
}

