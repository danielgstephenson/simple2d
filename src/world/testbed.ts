import { Brain } from '../brain'
import { Agent } from '../entities/agent'
import { World } from './world'

export class Testbed extends World {
  brain = new Brain()
  player: Agent
  bot: Agent
  timeScale = 1.4
  timeStep = 0.02

  constructor () {
    super()
    this.player = this.addAgent([-3, 0])
    this.bot = this.addAgent([+3, 0])
    this.summary = this.summarize()
    this.begin()
  }

  postStep (): void {
    this.summary = this.summarize()
    const state = this.getState()
    void this.brain.update(state)
  }

  preStep (): void {
    this.bot.action = this.brain.action
  }

  getState (): number[] {
    const playerState = this.player.getState()
    const botState = this.bot.getState()
    return [...botState, ...playerState]
  }
}
