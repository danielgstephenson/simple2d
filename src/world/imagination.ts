import { Agent } from '../entities/agent'
import { range } from '../math'
import { World } from './world'

export class Imagination extends World {
  agent0: Agent
  agent1: Agent
  stepCount = 10

  constructor () {
    super()
    this.agent0 = this.addAgent([-3, 0])
    this.agent1 = this.addAgent([+3, 0])
  }

  getOutcomes (state: number[]): number[][] {
    const state0 = range(0, 7).map(i => state[i])
    const state1 = range(8, 15).map(i => state[i])
    const outcomes: number[][] = []
    range(9).forEach(a0 => {
      range(9).forEach(a1 => {
        this.agent0.action = a0
        this.agent1.action = a1
        this.agent0.setState(state0)
        this.agent1.setState(state1)
        range(this.stepCount).forEach(_ => this.step())
        const outcome0 = this.agent0.getState()
        const outcome1 = this.agent1.getState()
        const outcome = [...outcome0, ...outcome1]
        outcomes.push(outcome)
      })
    })
    return outcomes
  }
}
