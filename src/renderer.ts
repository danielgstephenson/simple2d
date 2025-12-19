import { Camera } from './camera'
import { Agent } from './entities/agent'
import { Arena, ArenaSummary } from './entities/arena'
import { Blade } from './entities/blade'
import { WallSummary } from './entities/wall'
import { combine, dirFromTo, getDistance, range, X, Y } from './math'
import { WorldSummary } from './world/world'

export class Renderer {
  camera = new Camera()
  canvas: HTMLCanvasElement
  context: CanvasRenderingContext2D
  summary: WorldSummary
  renderScale = 1
  backgroundColor = 'hsl(0,0%,0%)'
  wallColor = 'hsl(0,0%,20%)'
  traceColor = 'hsla(0, 0%, 10%, 1.00)'
  springColor = 'hsla(0, 100%, 100%, 0.1)'
  agentColors = ['hsla(220,100%, 45%, 1.0)', 'hsla(120, 100%, 30%, 1.0)']
  bladeColors = ['hsla(220, 50%, 40%, 0.5)', 'hsla(120, 100%, 25%, 0.5)']

  constructor () {
    this.summary = {
      walls: [],
      blades: [],
      agents: [],
      arena: { boundary: [] }
    }
    this.canvas = document.getElementById('canvas') as HTMLCanvasElement
    this.context = this.canvas.getContext('2d') as CanvasRenderingContext2D
    this.draw()
  }

  draw (): void {
    window.requestAnimationFrame(() => this.draw())
    this.setupCanvas()
    this.followPlayer()
    this.drawArena(this.summary.arena)
    const agentCount = this.summary.agents.length
    range(agentCount).forEach(i => this.drawSpring(i))
    range(agentCount).forEach(i => this.drawBlade(i))
    range(agentCount).forEach(i => this.drawAgent(i))
    this.summary.walls.forEach(w => this.drawWall(w))
  }

  drawArena (arena: ArenaSummary): void {
    const boundary = arena.boundary
    this.context.fillStyle = this.wallColor
    this.context.fillRect(0, 0, this.canvas.width, this.canvas.height)
    this.resetContext()
    this.context.imageSmoothingEnabled = false
    this.context.fillStyle = this.backgroundColor
    this.context.beginPath()
    boundary.forEach((vertex, i) => {
      if (i === 0) this.context.moveTo(vertex[X], vertex[Y])
      else this.context.lineTo(vertex[X], vertex[Y])
    })
    this.context.fill()
    this.context.save()
    this.context.clip()
    this.context.strokeStyle = this.traceColor
    this.context.lineCap = 'round'
    this.context.lineWidth = 0.2
    this.context.beginPath()
    this.context.arc(0, 0, 5, 0, 2 * Math.PI)
    this.context.arc(0, 0, 10, 0, 2 * Math.PI)
    this.context.arc(0, 0, 20, 0, 2 * Math.PI)
    this.context.moveTo(0, Arena.size)
    this.context.lineTo(0, -Arena.size)
    this.context.moveTo(Arena.size, 0)
    this.context.lineTo(-Arena.size, 0)
    this.context.stroke()
  }

  drawSpring (i: number): void {
    this.resetContext()
    this.context.strokeStyle = this.springColor
    this.context.lineWidth = 0.08
    const blade = this.summary.blades[i]
    const agent = this.summary.agents[i]
    const distance = getDistance(blade.position, agent.position)
    if (distance < Blade.radius + Agent.radius) return
    const dir = dirFromTo(blade.position, agent.position)
    const edgePoint = combine(1, blade.position, Blade.radius, dir)
    this.context.lineCap = 'butt'
    this.context.beginPath()
    this.context.moveTo(edgePoint[X], edgePoint[Y])
    this.context.lineTo(agent.position[X], agent.position[Y])
    this.context.stroke()
  }

  drawBlade (i: number): void {
    this.resetContext()
    const blade = this.summary.blades[i]
    this.context.fillStyle = this.bladeColors[i]
    this.context.beginPath()
    this.context.arc(blade.position[X], blade.position[Y], Blade.radius, 0, 2 * Math.PI)
    this.context.fill()
  }

  drawAgent (i: number): void {
    this.resetContext()
    const agent = this.summary.agents[i]
    this.context.fillStyle = this.agentColors[i]
    this.context.beginPath()
    this.context.arc(agent.position[X], agent.position[Y], Agent.radius, 0, 2 * Math.PI)
    this.context.fill()
  }

  drawWall (wall: WallSummary): void {
    this.resetContext()
    this.context.strokeStyle = 'hsla(0, 0%, 50%, 1)'
    this.context.lineWidth = 0.1
    this.context.beginPath()
    this.context.moveTo(wall.a[X], wall.a[Y])
    this.context.lineTo(wall.b[X], wall.b[Y])
    this.context.stroke()
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
    this.context.translate(-this.camera.position[X], -this.camera.position[Y])
    this.context.globalAlpha = 1
  }
}
