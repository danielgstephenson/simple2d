import { Camera } from './camera'
import { CircleSummary } from './entities/circle'
import { WallSummary } from './entities/wall'
import { X, Y } from './math'
import { WorldSummary } from './world/world'

export class Renderer {
  camera = new Camera()
  canvas: HTMLCanvasElement
  context: CanvasRenderingContext2D
  summary: WorldSummary
  renderScale = 1

  backgroundColor = 'hsl(0,0%,0%)'
  wallColor = 'hsl(0,0%,35%)'

  constructor () {
    this.summary = {
      walls: [],
      circles: []
    }
    this.canvas = document.getElementById('canvas') as HTMLCanvasElement
    this.context = this.canvas.getContext('2d') as CanvasRenderingContext2D
    this.draw()
  }

  draw (): void {
    window.requestAnimationFrame(() => this.draw())
    this.setupCanvas()
    this.followPlayer()
    this.summary.circles.forEach(c => this.drawCircle(c))
    this.summary.walls.forEach(w => this.drawWall(w))
  }

  drawCircle (circle: CircleSummary): void {
    this.resetContext()
    this.context.fillStyle = 'hsl(220,100%,40%)'
    this.context.beginPath()
    this.context.arc(circle.position[X], circle.position[Y], circle.radius, 0, 2 * Math.PI)
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
    if (this.summary.circles.length === 0) {
      this.camera.position = [0, 0]
      return
    }
    this.camera.position = this.summary.circles[0].position
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
