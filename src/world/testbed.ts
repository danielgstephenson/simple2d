import { World } from './world'

export class Testbed extends World {
  constructor () {
    super()
    this.addAgent([-1, 0])
    this.addAgent([+1, 0])
    this.summary = this.summarize()
  }
}
