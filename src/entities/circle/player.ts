import { World } from '../../world/world'
import { Agent, AgentSummary } from './agent'
import { Star } from '../star'

export class Player extends Agent {
  static radius = 0.5
  align = 0

  constructor(world: World, position = [0, 0]) {
    super(world, position)
    world.players.push(this)
  }

  summarize(): AgentSummary {
    const summary = super.summarize()
    summary.spawnPoint = this.spawnPoint
    return summary
  }

  takeStar(star: Star): void {
    if (this.star != null) return
    if (star.agent != null) return
    this.star = star
    star.agent = this
  }

  respawn(): void {
    this.dead = false
    this.position = this.spawnPoint
    this.velocity = [0, 0]
    if (this.blade != null) {
      this.blade.detach()
    }
    if (this.blade != null) this.blade.detach()
    if (this.star != null) this.star.reset()
  }
}
