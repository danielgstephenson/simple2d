import { Brain } from '../brain'
import { Agent } from '../entities/agent/agent'
import { World } from './world'

export class TestArena extends World {
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

  preStep (): void {
    this.bot.action = this.brain.action
  }

  postStep (): void {
    this.summary = this.summarize()
    const state = this.getState()
    void this.brain.update(state)
    if (this.player.dead) {
      this.paused = true
    }
  }

  getState (): number[] {
    const playerState = this.player.getState()
    const botState = this.bot.getState()
    // For Pursuit:
    // const origin = [playerState[0], playerState[1]]
    // for (const i of range(8)) {
    //   const j = i % 2
    //   playerState[i] = playerState[i] - origin[j]
    //   botState[i] = botState[i] - origin[j]
    // }
    return [...botState, ...playerState]
  }
}
