import { combine, normalize } from '../math'
import { World } from '../world/world'
import { Blade } from './blade'
import { Circle } from './circle'

export class Agent extends Circle {
  static radius = 0.5
  blade?: Blade
  drag = 0.7
  movePower = 3
  velocity = [0, 0]
  force = [0, 0]
  action = 0
  dead = false

  constructor (world: World, position = [0, 0]) {
    super(world, position, Agent.radius)
    this.world.agents.push(this)
    this.blade = new Blade(this, position)
  }

  respawn (): void {
    this.dead = false
    const dir = normalize(this.position)
    this.position = combine(1, this.position, 15, dir)
    this.velocity = [0, 0]
    if (this.blade != null) {
      this.blade.position = structuredClone(this.position)
      this.blade.velocity = [0, 0]
    }
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
}
