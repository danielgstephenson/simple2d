import { Level } from './level'

export class TestLevel extends Level {

  constructor() {
    super()
    this.boundary = [
      [-10, -30],
      [10, -10],
      [25, 35],
      [-25, 20]
    ]
    this.walls.push([
      [30, -10],
      [-30, 40],
      [10, 10]
    ])
    this.addDoor([0, -5], [
      [-10, 10],
      [-10, 13],
      [-14, 13],
      [-14, 10]
    ])
    this.addTransporter([4, 4], [13, 13])
    this.addTransporter([2, 23], [-3, 13])
    this.addPlayerBlade([0, 10])
    this.addGuard([10, 0])
    this.addGuard([-10, 0])
    this.addGuardBlade([-10, 0])
    this.addStar([0, -10])
  }
}