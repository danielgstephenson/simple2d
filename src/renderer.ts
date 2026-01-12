import { Camera } from './camera'
import { Agent, AgentSummary } from './entities/agent/agent'
import { Blade, BladeSummary } from './entities/blade'
import { Circle } from './entities/circle'
import { Star, StarSummary } from './entities/star'
import { combine, dirFromTo, getDistance, pi, range } from './math'
import { WorldSummary } from './world/world'

export class Renderer {
  camera = new Camera()
  canvas: HTMLCanvasElement
  context: CanvasRenderingContext2D
  summary: WorldSummary
  renderScale = 1
  floorColor = 'hsl(0,0%,0%)'
  wallColor = 'hsl(0,0%,4%)'
  transportColor = 'hsl(0, 0%, 10%)'
  starColor = 'hsl(60, 100%, 40%)'
  agentColors = [
    'hsla(180, 50%, 25%, 1.0)',
    'hsla(220, 100%, 45%, 1.0)',
    'hsla(120, 100%, 30%, 1.0)']

  bladeColors = [
    'hsla(180, 40%, 20%, 0.5)',
    'hsla(220, 80%, 40%, 0.5)',
    'hsla(120, 100%, 25%, 0.5)']

  constructor () {
    this.summary = {
      boundary: [],
      walls: [],
      blades: [],
      agents: [],
      stars: []
    }
    this.canvas = document.getElementById('canvas') as HTMLCanvasElement
    this.context = this.canvas.getContext('2d') as CanvasRenderingContext2D
    this.draw()
  }

  draw (): void {
    window.requestAnimationFrame(() => this.draw())
    this.setupCanvas()
    this.followPlayer()
    this.drawBoundary(this.summary.boundary)
    this.summary.walls.forEach(w => this.drawWall(w))
    this.summary.blades.forEach(blade => this.drawSpring(blade))
    this.summary.blades.forEach(blade => this.drawBlade(blade))
    this.summary.agents.forEach(agent => this.drawAgent(agent))
    this.summary.stars.forEach(star => this.drawStar(star))
  }

  drawBoundary (boundary: number[][]): void {
    this.context.fillStyle = this.wallColor
    this.context.fillRect(0, 0, this.canvas.width, this.canvas.height)
    this.resetContext()
    this.context.imageSmoothingEnabled = false
    this.context.fillStyle = this.floorColor
    this.context.beginPath()
    boundary.forEach((vertex, i) => {
      if (i === 0) this.context.moveTo(vertex[0], vertex[1])
      else this.context.lineTo(vertex[0], vertex[1])
    })
    this.context.fill()
    this.context.save()
    this.context.clip()
    this.context.strokeStyle = this.transportColor
    this.context.lineCap = 'round'
    this.context.lineWidth = 0.03
    const size = 5
    const points = [
      [0, size],
      [0, -size],
      [size, 0],
      [-size, 0],
      [+size * Math.SQRT1_2, +size * Math.SQRT1_2],
      [-size * Math.SQRT1_2, -size * Math.SQRT1_2],
      [+size * Math.SQRT1_2, -size * Math.SQRT1_2],
      [-size * Math.SQRT1_2, +size * Math.SQRT1_2]
    ]
    this.context.beginPath()
    this.context.arc(0, 0, size, 0, 2 * Math.PI)
    for (const a of points) {
      for (const b of points) {
        this.context.moveTo(a[0], a[1])
        this.context.lineTo(b[0], b[1])
      }
    }
    this.context.stroke()
  }

  drawWall (wall: number[][]): void {
    this.resetContext()
    this.context.fillStyle = this.wallColor
    this.context.lineWidth = 0.1
    this.context.beginPath()
    this.context.beginPath()
    wall.forEach((point, i) => {
      if (i === 0) this.context.moveTo(point[0], point[1])
      else this.context.lineTo(point[0], point[1])
    })
    this.context.fill()
  }

  drawStar (star: StarSummary): void {
    this.resetContext()
    this.context.fillStyle = this.starColor
    this.context.beginPath()
    const origin = star.agent == null
      ? star.spawnPoint
      : this.summary.agents[star.agent].position
    range(5).forEach(i => {
      const angle0 = (0.5 + 0.4 * i) * pi
      const x0 = origin[0] + Math.cos(angle0) * Star.radius
      const y0 = origin[1] + Math.sin(angle0) * Star.radius
      if (i === 0) this.context.moveTo(x0, y0)
      else this.context.lineTo(x0, y0)
      const angle1 = angle0 + 0.2 * pi
      const x1 = origin[0] + Math.cos(angle1) * Star.radius * 0.45
      const y1 = origin[1] + Math.sin(angle1) * Star.radius * 0.45
      this.context.lineTo(x1, y1)
    })
    // this.context.arc(star.spawnPoint[0], star.spawnPoint[1], Star.radius, 0, 2 * Math.PI)
    this.context.fill()
  }

  drawSpring (blade: BladeSummary): void {
    if (blade.agent == null) return
    const agent = this.summary.agents[blade.agent]
    this.resetContext()
    this.context.strokeStyle = this.bladeColors[blade.align]
    this.context.lineWidth = 0.08
    const distance = getDistance(blade.position, agent.position)
    if (distance < Blade.radius + Agent.radius) return
    const dir = dirFromTo(blade.position, agent.position)
    const edgePoint = combine(1, blade.position, Blade.radius, dir)
    this.context.lineCap = 'butt'
    this.context.beginPath()
    this.context.moveTo(edgePoint[0], edgePoint[1])
    this.context.lineTo(agent.position[0], agent.position[1])
    this.context.stroke()
  }

  drawBlade (blade: BladeSummary): void {
    this.resetContext()
    const L = Circle.historyLength
    blade.history.forEach((position, i) => {
      const a = 0.01 * (L - i) / L
      this.context.fillStyle = `hsla(0, 0%, 50%, ${a})`
      this.context.beginPath()
      this.context.arc(position[0], position[1], Blade.radius, 0, 2 * Math.PI)
      this.context.fill()
    })
    this.context.fillStyle = this.bladeColors[blade.align]
    this.context.beginPath()
    this.context.arc(blade.position[0], blade.position[1], Blade.radius, 0, 2 * Math.PI)
    this.context.fill()
  }

  drawAgent (agent: AgentSummary): void {
    if (agent.dead && agent.align !== 1) return
    this.resetContext()
    const L = Circle.historyLength
    agent.history.forEach((position, i) => {
      const a = 0.02 * (L - i) / L
      this.context.fillStyle = `hsla(0, 0%, 50%, ${a})`
      this.context.beginPath()
      this.context.arc(position[0], position[1], Agent.radius, 0, 2 * Math.PI)
      this.context.fill()
    })
    this.context.fillStyle = this.agentColors[agent.align]
    this.context.beginPath()
    this.context.arc(agent.position[0], agent.position[1], Agent.radius, 0, 2 * Math.PI)
    this.context.fill()

    this.context.strokeStyle = 'red'
    this.context.lineWidth = 0.1
  }

  followPlayer (): void {
    if (this.summary.agents.length === 0) {
      this.camera.position = [0, 0]
      return
    }
    this.camera.position = this.summary.agents[0].position
  }

  setupCanvas (): void {
    this.canvas.width = window.innerWidth * this.renderScale
    this.canvas.height = window.innerHeight * this.renderScale
    // this.context.imageSmoothingEnabled = false
  }

  resetContext (): void {
    this.context.resetTransform()
    this.context.translate(0.5 * this.canvas.width, 0.5 * this.canvas.height)
    const vmin = Math.min(this.canvas.width, this.canvas.height)
    this.context.scale(vmin, -vmin)
    this.context.scale(this.camera.scale, this.camera.scale)
    this.context.translate(-this.camera.position[0], -this.camera.position[1])
    this.context.globalAlpha = 1
  }
}
