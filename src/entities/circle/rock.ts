import { World } from '../../world/world'
import { Circle } from './circle'

export class Rock extends Circle {
  drag = 2
  velocity = [0, 0]
  force = [0, 0]
  action = 0
  dead = false
  align = 1
  radius: number
  index: number

  constructor(world: World, position = [0, 0], radius: number) {
    super(world, position, radius)
    this.radius = radius
    this.index = world.rocks.length
    this.world.rocks.push(this)
  }

  getState(): number[] {
    const ap = this.position
    const av = this.velocity
    return [...ap, ...av]
  }

  setState(state: number[]): void {
    this.position = [state[0], state[1]]
    this.velocity = [state[2], state[3]]
  }

  summarize(): RockSummary {
    return {
      position: this.position,
      radius: this.radius
    }
  }
}

export interface RockSummary {
  position: number[]
  radius: number
}
