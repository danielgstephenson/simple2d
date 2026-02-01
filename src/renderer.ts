import { Camera } from './camera'
import { Agent, AgentSummary } from './entities/agent/agent'
import { Blade, BladeSummary } from './entities/blade'
import { Circle } from './entities/circle'
import { DoorSummary } from './entities/door'
import { Star, StarSummary } from './entities/star'
import { Transporter, TransporterSummary } from './entities/transporter'
import { angleToDir, combine, dirFromTo, getDistance, pi, range } from './math'
import { Layout, LevelSummary } from './world/level'


export class Renderer {
  camera = new Camera()
  canvas: HTMLCanvasElement
  context: CanvasRenderingContext2D
  summary: LevelSummary
  layout: Layout
  renderScale = 1
  floorColor = 'hsl(0,0%,6%)'
  wallColor = 'hsl(0,0%,0%)'
  transportColor = 'hsla(0, 0%, 100%, 0.3)'
  doorColor = 'hsl(36, 100%, 6%)'
  starColor = 'hsl(60, 100%, 40%)'
  agentColors = [
    'hsl(220, 100%, 50%)',
    'hsl(120, 100%, 35%)'
  ]
  bladeColors = [
    'hsl(220, 100%, 25%)',
    'hsl(120, 100%, 15%)']

  constructor() {
    this.summary = {
      blades: [],
      agents: [],
      stars: [],
      doors: [],
      transporters: []
    }
    this.layout = {
      boundary: [],
      walls: [],
      floorPoints: []
    }
    this.canvas = document.getElementById('canvas') as HTMLCanvasElement
    this.context = this.canvas.getContext('2d') as CanvasRenderingContext2D
    this.draw()
  }

  draw(): void {
    window.requestAnimationFrame(() => this.draw())
    this.context.save()
    this.setupCanvas()
    this.followPlayer()
    this.drawBoundary(this.layout.boundary)
    this.drawFloor(this.layout)
    this.summary.transporters.forEach(transporter => this.drawTransporter(transporter))
    this.summary.doors.forEach(door => this.drawDoor(door))
    this.summary.blades.forEach(blade => this.drawSpring(blade))
    this.summary.blades.forEach(blade => this.drawBlade(blade))
    this.summary.agents.forEach(agent => this.drawAgent(agent))
    this.summary.stars.forEach(star => this.drawStar(star))
    this.layout.walls.forEach(w => this.drawWall(w))
    this.context.restore()
  }

  drawBoundary(boundary: number[][]): void {
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
    this.context.clip()
  }

  drawTransporter(transporter: TransporterSummary): void {
    this.context.strokeStyle = this.transportColor
    this.context.lineCap = 'round'
    this.context.lineWidth = 0.05
    const shell = range(8).map(i => {
      const angle = i / 8 * 2 * pi
      const dir = angleToDir(angle)
      return combine(1, transporter.center, Transporter.radius, dir)
    })
    this.context.beginPath()
    const x = transporter.center[0]
    const y = transporter.center[1]
    this.context.arc(x, y, Transporter.radius, 0, 2 * Math.PI)
    for (const a of shell) {
      for (const b of shell) {
        this.context.moveTo(a[0], a[1])
        this.context.lineTo(b[0], b[1])
      }
    }
    this.context.stroke()
    if (transporter.charge === 0) return
    this.context.strokeStyle = this.transportColor
    this.context.lineWidth = 0.5
    this.context.globalAlpha = 0.25
    const radius = Transporter.radius + 0.5
    const start = Math.PI * (0.5 - 2 * transporter.charge / transporter.interval)
    const stop = 0.5 * Math.PI
    this.context.beginPath()
    this.context.arc(x, y, radius, start, stop)
    this.context.stroke()
  }

  drawFloor(layout: Layout): void {
    this.context.strokeStyle = this.transportColor
    this.context.lineCap = 'round'
    this.context.lineWidth = 0.03
    layout.floorPoints.forEach((point) => {
      const lightness = 4
      const x = point[0]
      const y = point[1]
      const radius = 4
      const gradient = this.context.createRadialGradient(x, y, 0, x, y, radius)
      gradient.addColorStop(0, `hsla(0,0%,${lightness}%,0.5)`)
      gradient.addColorStop(1, `hsla(0,0%,${lightness}%,0)`)
      this.context.fillStyle = gradient
      this.context.beginPath()
      this.context.arc(x, y, radius, 0, 2 * Math.PI)
      this.context.fill()
    })
  }

  drawWall(wall: number[][]): void {
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

  drawDoor(door: DoorSummary): void {
    const shell = door.shell
    this.resetContext()
    this.context.fillStyle = this.doorColor
    this.context.lineWidth = 0.1
    this.context.beginPath()
    const xs: number[] = []
    const ys: number[] = []
    shell.forEach((point, i) => {
      if (i === 0) this.context.moveTo(point[0], point[1])
      else this.context.lineTo(point[0], point[1])
      xs.push(point[0])
      ys.push(point[1])
    })
    this.context.fill()
    this.context.strokeStyle = this.starColor
    this.context.lineWidth = 0.05
    this.context.beginPath()
    const origin = door.center
    const radius = Star.radius + this.context.lineWidth
    range(5).forEach(i => {
      const angle0 = (0.5 + 0.4 * i) * pi
      const x0 = origin[0] + Math.cos(angle0) * radius
      const y0 = origin[1] + Math.sin(angle0) * radius
      if (i === 0) this.context.moveTo(x0, y0)
      else this.context.lineTo(x0, y0)
      const angle1 = angle0 + 0.2 * pi
      const x1 = origin[0] + Math.cos(angle1) * radius * 0.45
      const y1 = origin[1] + Math.sin(angle1) * radius * 0.45
      this.context.lineTo(x1, y1)
    })
    this.context.closePath()
    this.context.stroke()
  }

  drawStar(star: StarSummary): void {
    this.resetContext()
    this.context.fillStyle = this.starColor
    this.context.beginPath()
    let origin = star.spawnPoint
    if (star.agent != null) {
      origin = this.summary.agents[star.agent].position
    }
    if (star.door != null) {
      origin = this.summary.doors[star.door].center
    }
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
    this.context.fill()
  }

  drawSpring(blade: BladeSummary): void {
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

  drawBlade(blade: BladeSummary): void {
    this.resetContext()
    const L = Circle.historyLength
    this.context.fillStyle = this.bladeColors[blade.align]
    this.context.globalCompositeOperation = 'lighten'
    blade.history.forEach((position, i) => {
      this.context.globalAlpha = 0.1 * (L - i) / L
      this.context.beginPath()
      this.context.arc(position[0], position[1], Blade.radius, 0, 2 * Math.PI)
      this.context.fill()
    })
    this.context.globalCompositeOperation = 'source-over'
    this.context.globalAlpha = 1
    this.context.beginPath()
    this.context.arc(blade.position[0], blade.position[1], Blade.radius, 0, 2 * Math.PI)
    this.context.fill()
  }

  drawAgent(agent: AgentSummary): void {
    if (agent.dead && agent.align !== 0) return
    this.resetContext()
    const L = Circle.historyLength
    this.context.fillStyle = this.agentColors[agent.align]
    this.context.globalCompositeOperation = 'lighten'
    agent.history.forEach((position, i) => {
      this.context.globalAlpha = 0.05 * (L - i) / L
      this.context.beginPath()
      this.context.arc(position[0], position[1], Agent.radius, 0, 2 * Math.PI)
      this.context.fill()
    })
    this.context.globalCompositeOperation = 'source-over'
    this.context.globalAlpha = 1
    this.context.beginPath()
    this.context.arc(agent.position[0], agent.position[1], Agent.radius, 0, 2 * Math.PI)
    this.context.fill()
  }

  followPlayer(): void {
    if (this.summary.agents.length === 0) {
      this.camera.position = [0, 0]
      return
    }
    this.camera.position = this.summary.agents[0].position
  }

  setupCanvas(): void {
    this.canvas.width = window.innerWidth * this.renderScale
    this.canvas.height = window.innerHeight * this.renderScale
    // this.context.imageSmoothingEnabled = false
  }

  resetContext(): void {
    this.context.resetTransform()
    this.context.translate(0.5 * this.canvas.width, 0.5 * this.canvas.height)
    const vmin = Math.min(this.canvas.width, this.canvas.height)
    this.context.scale(vmin, -vmin)
    this.context.scale(this.camera.scale, this.camera.scale)
    this.context.translate(-this.camera.position[0], -this.camera.position[1])
    this.context.globalAlpha = 1
  }
}
