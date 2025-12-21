import { range } from '../math'
import { World } from './world'

export class Testbed extends World {
  constructor () {
    super()
    const agent1 = this.addAgent([-3, 0])
    const agent2 = this.addAgent([+3, 0])
    agent1.velocity = [+1, 0]
    agent2.velocity = [-1, 0]
    agent1.blade.position = [-2, -4]
    agent2.blade.position = [+2, +4]
    agent1.blade.velocity = [+5, 0]
    agent2.blade.velocity = [-5, 0]
    this.summary = this.summarize()

    range(50).forEach(step => {
      console.log(step)
      this.agents.forEach(agent => {
        console.log(agent.position)
      })
      this.step()
    })
  }
}
