import NanoTimer from 'nanotimer'
import { actionVectors } from '../actionVectors'
import { Agent } from '../entities/agent/agent'
import { Player } from '../entities/agent/player'
import { Blade } from '../entities/blade'
import { Circle } from '../entities/circle'
import { Door } from '../entities/door'
import { Star } from '../entities/star'
import { Transporter } from '../entities/transporter'
import { add, clamp, clampVec, combine, dirFromTo, dot, getDistance, mul, normalize, range, sub, X, Y } from '../math'
import { Rock } from '../entities/rock'

export class World {
  timer = new NanoTimer()
  rocks: Rock[] = []
  agents: Agent[] = []
  players: Player[] = []
  blades: Blade[] = []
  stars: Star[] = []
  circles: Circle[] = []
  doors: Door[] = []
  walls: number[][][] = []
  boundary: number[][] = []
  transporters: Transporter[] = []
  timeStep = 0.02
  timeScale = 1
  time = 0
  busy = false
  paused = false

  constructor() {
    this.timer.setInterval(() => this.step(), '', `${this.timeStep / this.timeScale}s`)
  }

  addPlayer(position: number[]): Player {
    const player = new Player(this, position)
    return player
  }

  addAgent(position: number[]): Agent {
    const agent = new Agent(this, position)
    return agent
  }

  addRock(position: number[], radius: number): Rock {
    const rock = new Rock(this, position, radius)
    return rock
  }

  addPlayerBlade(position: number[]): Blade {
    const blade = new Blade(this, position)
    blade.align = 0
    return blade
  }

  addGuardBlade(position: number[]): Blade {
    const blade = new Blade(this, position)
    blade.align = 1
    return blade
  }

  addDoor(vector: number[], shell: number[][]): Door {
    const door = new Door(this, vector, shell)
    return door
  }

  addStar(position: number[]): Star {
    const star = new Star(this, position)
    return star
  }

  addTransporter(position: number[], target: number[]): Transporter {
    const transporter = new Transporter(this, position, target)
    return transporter
  }

  preStep(): void { }
  postStep(): void { }

  step(): void {
    if (this.busy) {
      console.log('busy')
      return
    }
    if (this.paused) return
    this.busy = true
    this.preStep()
    const dt = this.timeStep
    this.time += dt
    this.circles.forEach(circle => {
      circle.force = [0, 0]
      circle.impulse = [0, 0]
      circle.shift = [0, 0]
    })
    this.agents.forEach(agent => {
      if (agent.dead) return
      agent.force = mul(agent.movePower, actionVectors[agent.action])
    })
    this.blades.forEach(blade => {
      if (blade.agent != null) {
        const vector = sub(blade.agent.position, blade.position)
        const clampedVector = clampVec(vector, 10)
        blade.force = mul(blade.movePower, clampedVector)
      }
    })
    this.doors.forEach(door => { door.onStep(dt) })
    this.transporters.forEach(transporter => {
      let chargeRate = -1
      this.players.forEach(player => {
        const distance = getDistance(player.position, transporter.center)
        if (distance > 1) return
        if (transporter.charge === transporter.interval) {
          transporter.transport(player)
          return
        }
        chargeRate = 1
      })
      transporter.charge = clamp(0, transporter.interval, transporter.charge + chargeRate * dt)
    })
    this.players.forEach(player => {
      this.stars.forEach(star => {
        if (star.agent != null) return
        if (star.door != null) return
        if (player.star != null) return
        const distance = getDistance(player.position, star.spawnPoint)
        if (distance > Star.radius + Agent.radius) return
        player.takeStar(star)
      })
      if (player.dead) {
        player.respawn()
      }
    })
    this.agents.forEach(agent => {
      if (agent.dead) return
      this.checkBlades(agent)
    })
    this.agents.forEach(agent1 => {
      if (agent1.dead) return
      this.agents.forEach(agent2 => {
        if (agent2.dead) return
        if (agent1.index >= agent2.index) return
        this.collideCircleCircle(agent1, agent2)
      })
    })
    this.rocks.forEach(rock => {
      this.agents.forEach(agent => {
        if (agent.dead) return
        this.collideCircleCircle(rock, agent)
      })
      this.rocks.forEach(rock2 => {
        this.collideCircleCircle(rock, rock2)
      })
    })
    this.blades.forEach(blade1 => {
      this.blades.forEach(blade2 => {
        if (blade1.index >= blade2.index) return
        this.collideCircleCircle(blade1, blade2)
      })
    })
    this.circles.forEach(circle => {
      this.collideCircleWall(circle, this.boundary)
      this.walls.forEach(wall => {
        this.collideCircleWall(circle, wall)
      })
      this.doors.forEach(door => {
        const hit = this.collideCircleWall(circle, door.shell)
        if (hit && circle instanceof Player) door.knock(circle)
      })
    })
    this.circles.forEach(circle => {
      circle.velocity = mul(1 - circle.drag * dt, circle.velocity)
      circle.velocity = combine(1, circle.velocity, dt / circle.mass, circle.force)
      circle.velocity = combine(1, circle.velocity, 1 / circle.mass, circle.impulse)
      circle.position = combine(1, circle.position, dt, circle.velocity)
      circle.position = combine(1, circle.position, 1, circle.shift)
    })
    this.circles.forEach(circle => {
      circle.history.unshift(circle.position)
      circle.history = circle.history.slice(0, Circle.historyLength)
    })
    this.postStep()
    this.busy = false
  }

  collideCircleCircle(circle1: Circle, circle2: Circle): boolean {
    const distance = getDistance(circle1.position, circle2.position)
    const overlap = circle1.radius + circle2.radius - distance
    if (overlap <= 0) return false
    const normal = dirFromTo(circle1.position, circle2.position)
    const relativeVelocity = sub(circle1.velocity, circle2.velocity)
    const impactSpeed = dot(relativeVelocity, normal)
    const massFactor = 1 / (1 / circle1.mass + 1 / circle2.mass)
    const impulse = mul(impactSpeed * massFactor, normal)
    const shift = mul(0.5 * overlap, normal)
    circle1.impulse = combine(1, circle1.impulse, -1, impulse)
    circle2.impulse = combine(1, circle2.impulse, +1, impulse)
    circle1.shift = combine(1, circle1.shift, -1, shift)
    circle2.shift = combine(1, circle2.shift, +1, shift)
    return true
  }

  collideCircleWall(circle: Circle, wall: number[][]): boolean {
    let hit = false
    for (const i of range(wall.length)) {
      const j = i > 0 ? i - 1 : wall.length - 1
      const a = wall[i]
      const b = wall[j]
      const c = circle.position
      const ab = sub(b, a)
      const ac = sub(c, a)
      const bc = sub(c, b)
      const dir = normalize(ab)
      const normal = [-dir[Y], +dir[X]]
      if (dot(normal, ac) < 0) {
        normal[X] = -normal[X]
        normal[Y] = -normal[Y]
      }
      if (dot(ac, ab) < 0) continue
      if (dot(bc, ab) > 0) continue
      const overlap = circle.radius - dot(ac, normal)
      if (overlap < 0) continue
      const impactSpeed = -dot(circle.velocity, normal)
      const impulse = mul(1.2 * impactSpeed * circle.mass, normal)
      circle.impulse = add(circle.impulse, impulse)
      const shift = mul(overlap, normal)
      circle.shift = add(circle.shift, shift)
      hit = true
    }
    if (hit) return hit
    for (const point of wall) {
      if (this.collideCirclePoint(circle, point)) {
        hit = true
      }
    }
    return hit
  }

  collideCirclePoint(circle: Circle, point: number[]): boolean {
    const distance = getDistance(circle.position, point)
    const overlap = circle.radius - distance
    if (overlap <= 0) return false
    const normal = dirFromTo(point, circle.position)
    const impactSpeed = -dot(circle.velocity, normal)
    const impulse = mul(1.2 * impactSpeed * circle.mass, normal)
    circle.impulse = add(circle.impulse, impulse)
    const shift = mul(overlap, normal)
    circle.shift = add(circle.shift, shift)
    return true
  }

  checkBlades(agent: Agent): void {
    if (agent.dead) return
    this.blades.forEach(blade => {
      if (agent.dead) return
      const distance = getDistance(agent.position, blade.position)
      const overlap = agent.radius + blade.radius - distance
      if (overlap < 0) return
      if (blade.align === agent.align) {
        if (agent.blade == null && blade.agent == null) blade.attach(agent)
        return
      }
      agent.die()
      return
    })
  }
}
