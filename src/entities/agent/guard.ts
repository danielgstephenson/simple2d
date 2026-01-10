import { World } from '../../world/world'
import { Agent } from './agent'

export class Guard extends Agent {
  static radius = 0.5
  align = 2

  constructor (world: World, position = [0, 0]) {
    super(world, position)
  }

  die (): void {
    super.die()
    if (this.blade != null) this.blade.detach()
    if (this.star != null) this.star.reset()
  }
}
