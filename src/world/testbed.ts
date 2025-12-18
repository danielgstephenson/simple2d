import { World } from './world'

export class Testbed extends World {
  constructor () {
    super()
    this.addCircle([0, 0], 0.5)
    this.addWall([-2, -2], [2, -2])
    this.summary = this.summarize()
  }
}
