import { randomDir } from '../../math'
import { World } from '../../world/world'
import { Blade } from '../blade'
import { Circle } from '../circle'
import { Star } from '../star'

export class Agent extends Circle {
  static radius = 0.5
  blade?: Blade
  star?: Star
  drag = 0.7
  movePower = 3
  spawnPoint = [0, 0]
  velocity = [0, 0]
  force = [0, 0]
  action = 0
  dead = false
  align = 0
  index: number
  rayDir = randomDir()
  rayPoints: number[][] = []

  constructor (world: World, position = [0, 0]) {
    super(world, position, Agent.radius)
    this.spawnPoint = position
    this.index = world.agents.length
    this.world.agents.push(this)
  }

  die (): void {
    this.dead = true
  }

  getState (): number[] {
    const ap = this.position
    const av = this.velocity
    const bp = this.blade != null ? this.blade.position : [0, 0]
    const bv = this.blade != null ? this.blade.velocity : [0, 0]
    return [...ap, ...av, ...bp, ...bv]
  }

  setState (state: number[]): void {
    this.position = [state[0], state[1]]
    this.velocity = [state[2], state[3]]
    if (this.blade != null) {
      this.blade.position = [state[4], state[5]]
      this.blade.velocity = [state[6], state[7]]
    }
  }

  summarize (): AgentSummary {
    return {
      position: this.position,
      history: this.history,
      align: this.align,
      dead: this.dead,
      blade: this.blade == null ? undefined : this.blade.index,
      rayPoints: this.rayPoints
    }
  }
}

export interface AgentSummary {
  position: number[]
  history: number[][]
  align: number
  dead: boolean
  blade?: number
  rayPoints: number[][]
}
