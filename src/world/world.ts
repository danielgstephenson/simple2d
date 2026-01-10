import { actionVectors } from '../actionVectors'
import { Agent, AgentSummary } from '../entities/agent'
import { Blade, BladeSummary } from '../entities/blade'
import { Circle } from '../entities/circle'
import { Star, StarSummary } from '../entities/star'
import { add, clampVec, combine, dirFromTo, dot, getDistance, mul, normalize, range, sub, X, Y } from '../math'

export class World {
  agents: Agent[] = []
  blades: Blade[] = []
  stars: Star[] = []
  circles: Circle[] = []
  walls: number[][][] = []
  boundary: number[][] = [] // This is the outer boundary
  summary: WorldSummary
  timeStep = 0.04
  timeScale = 1
  paused = false

  constructor () {
    this.summary = this.summarize()
  }

  begin (): void {
    setInterval(() => this.step(), 1000 * this.timeStep / this.timeScale)
  }

  addAgent (position: number[]): Agent {
    const agent = new Agent(this, position)
    return agent
  }

  addBlade (position: number[]): Blade {
    const blade = new Blade(this, position)
    return blade
  }

  addStar (position: number[]): Star {
    const star = new Star(this, position)
    return star
  }

  summarize (): WorldSummary {
    return {
      boundary: this.boundary,
      walls: this.walls,
      agents: this.agents.map(c => c.summarize()),
      blades: this.blades.map(b => b.summarize()),
      stars: this.stars.map(s => s.summarize())
    }
  }

  preStep (): void {}
  postStep (): void {}

  step (): void {
    if (this.paused) return
    this.preStep()
    const dt = this.timeStep
    this.agents.forEach(agent => {
      agent.force = [0, 0]
      agent.impulse = [0, 0]
      agent.shift = [0, 0]
    })
    this.blades.forEach(blade => {
      blade.force = [0, 0]
      blade.impulse = [0, 0]
      blade.shift = [0, 0]
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
  }

  collideCircleCircle (circle1: Circle, circle2: Circle): boolean {
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

  collideCircleWall (circle: Circle, wall: number[][]): void {
    for (const point of wall) {
      if (this.collideCirclePoint(circle, point)) {
        return
      }
    }
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
      const impulse = mul(2 * impactSpeed * circle.mass, normal)
      circle.impulse = add(circle.impulse, impulse)
      const shift = mul(overlap, normal)
      circle.shift = add(circle.shift, shift)
    }
  }

  collideCirclePoint (circle: Circle, point: number[]): boolean {
    const distance = getDistance(circle.position, point)
    const overlap = circle.radius - distance
    if (overlap <= 0) return false
    const normal = dirFromTo(point, circle.position)
    const impactSpeed = -dot(circle.velocity, normal)
    const impulse = mul(2 * impactSpeed * circle.mass, normal)
    circle.impulse = add(circle.impulse, impulse)
    const shift = mul(overlap, normal)
    circle.shift = add(circle.shift, shift)
    return true
  }

  checkBlades (agent: Agent): void {
    if (agent.dead) return
    this.blades.forEach(blade => {
      if (agent.dead) return
      const distance = getDistance(agent.position, blade.position)
      const overlap = agent.radius + blade.radius - distance
      if (overlap < 0) return
      if (blade.align === 0 || blade.align === agent.align) {
        if (agent.blade == null && blade.agent == null) blade.attach(agent)
        return
      }
      agent.die()
    })
  }
}

export interface WorldSummary {
  agents: AgentSummary[]
  blades: BladeSummary[]
  stars: StarSummary[]
  walls: number[][][]
  boundary: number[][]
}
