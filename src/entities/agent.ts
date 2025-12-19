import { combine, normalize } from '../math'
import { World } from '../world/world'
import { Blade } from './blade'
import { Circle } from './circle'

export class Agent extends Circle {
  static radius = 0.5
  blade: Blade
  drag = 0.7
  movePower = 3
  velocity = [0, 0]
  force = [0, 0]
  action = 0

  constructor (world: World, position = [0, 0]) {
    super(world, position, Agent.radius)
    this.world.agents.push(this)
    this.blade = new Blade(world, position)
  }

  die (): void {
    const dir = normalize(this.position)
    this.position = combine(1, this.position, 15, dir)
    this.blade.position = structuredClone(this.position)
    this.velocity = [0, 0]
    this.blade.velocity = [0, 0]
  }
}
