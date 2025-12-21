import { Agent } from './agent'
import { Circle } from './circle'

export class Blade extends Circle {
  static radius = 1
  agent: Agent
  drag = 0.3
  movePower = 4
  velocity = [0, 0]
  force = [0, 0]
  collideForce = [0, 0]

  constructor (agent: Agent, position = [0, 0]) {
    super(agent.world, position, Blade.radius)
    this.agent = agent
    this.world.blades.push(this)
  }
}
